#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Mon Mar  5 09:34:11 2018



@author: ljx

"""

import os; os.environ["TF_KERAS"]='1'


from comet_ml import Experiment

#create an experiment with your api key
experiment = Experiment(
    api_key="z2kBCBeTsLxTn7mHU2YEuCdcI",
    project_name="general",
    workspace="bognev",
    auto_metric_logging=True,
    auto_param_logging=True,
    auto_histogram_weight_logging=True,
    auto_histogram_gradient_logging=True,
    auto_histogram_activation_logging=True,
    log_graph=True,
)
import matplotlib.pyplot as plt
#==============================================================================
#train the network with bidirectional lstm, train the deltas of trajectory
#==============================================================================
import numpy as np
import tensorflow as tf
# tf.autograph.set_verbosity(2, True)
from tensorflow.python.ops import variable_scope as vs
from batchdata_derive3 import *
from models import *
from comet_help import *

from datetime import datetime

# Disable V2 behavior
# tf.compat.v1.disable_v2_behavior()

SEED = 2021
base_path = './Models'

def set_seed(seed=42):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(SEED)



print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(False)


#==============================================================================
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))





#==============================================================================

# boundaries = [1000, 3000]
# values = [0.01, 0.001, 0.0001]
# learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.75)

opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_fn)
loss_fn = root_mean_squared_error
epoch_loss_metric = tf.keras.metrics.MeanAbsoluteError()
val_loss_metric = tf.keras.metrics.MeanAbsoluteError()

# The size of batch for learning
batch_size = 32
#x,y,vx,vy
input_size = 4
#Time sequence duration
timestep_size = 64

#Number of hidden layers
# LSTM layer
layer_num = 3
hidden_size_1 = 128
hidden_size_2 = 256
hidden_size_3 = 256
maxout_size = 64
#regularization
lambda1 = 5
#FIR samples
fir_size = 5
# The dimension of the final output vector
output_size = 4
keep_prob = 0.0
PLOT = 1

epochs = 2048
steps_per_epoch = 16

params={'batch_size':batch_size,
        'epochs':epochs,
        'layer1_type':'Bidirectional',
        'layer1_num_nodes': hidden_size_1,
        'layer1_activation': 'tanh',
        'optimizer': opt
}

init_lr = 0.01



#==============================================================================
if __name__ == "__main__":
    # plot_trajectory()
    # plt.show()

    # fold=0
    experiment.log_parameters(params)
    gradmap = {}
    activations = {}
    model = create_gmodel(batch_size, timestep_size, input_size, keep_prob)
    print(model.summary())
    names = [weight.name for layer in model.layers for weight in layer.weights]
    print(names)
    generator = make_generator(batch_size, timestep_size)
    # X_val, y_val = next(generator)
    X_train, y_train = next(generator)
    with experiment.train():
        for epoch in range(epochs):
            X, y = next(generator)
            for step in range(steps_per_epoch):
                permutation = np.random.permutation(batch_size)
                X, y = X[permutation], y[permutation]

                with tf.GradientTape() as tape:
                    preds = model(X, training=True)
                    loss = loss_fn(y, preds)

                gradients = tape.gradient(loss, model.trainable_weights)
                opt.apply_gradients(zip(gradients, model.trainable_weights))

                gradmap = get_gradients(gradmap, gradients, model)
                activations = get_activations(activations, X, model)

                experiment.log_metric("batch_loss", loss, step=step + epoch*steps_per_epoch)
                epoch_loss_metric.update_state(y, preds)

            epoch_loss = epoch_loss_metric.result()
            print("Epoch:", epoch, "step:", step, "Training loss over epoch: %.4f" % (float(epoch_loss),))
            experiment.log_metric("epoch_loss", epoch_loss, step=epoch)
            experiment.log_metric("learning_rate", opt.learning_rate(opt.iterations).numpy(), step=epoch)
            # Reset training metrics at the end of each epoch
            epoch_loss_metric.reset_states()

            # scale gradients
            for k, v in gradmap.items():
                gradmap[k] = v / steps_per_epoch

            # scale activations
            for k, v in activations.items():
                activations[k] = v / steps_per_epoch

            log_weights(experiment, model, step=epoch)
            log_histogram(experiment, gradmap, step=epoch, prefix="gradient")
            log_histogram(experiment, activations, step=epoch, prefix="activation")
            # experiment.log_histogram_3d(preds.numpy(), name="preds")
            # experiment.log_curve("pred res Dx", x=np.arange(timestep_size).tolist(), y=preds[0, :, 0].numpy().tolist(), overwrite=True)
            # experiment.log_curve("pred res Dy", x=np.arange(timestep_size).tolist(), y=preds[0, :, 1].numpy().tolist(), overwrite=True)
            # experiment.log_curve("pred res Vx", x=np.arange(timestep_size).tolist(), y=preds[0, :, 2].numpy().tolist(), overwrite=True)
            # experiment.log_curve("pred res Vy", x=np.arange(timestep_size).tolist(), y=preds[0, :, 3].numpy().tolist(), overwrite=True)
            #
            # experiment.log_curve("res Dx", x=np.arange(timestep_size).tolist(), y=y[0, :, 0].tolist(),overwrite=True)
            # experiment.log_curve("res Dy", x=np.arange(timestep_size).tolist(), y=y[0, :, 1].tolist(),overwrite=True)
            # experiment.log_curve("res Vx", x=np.arange(timestep_size).tolist(), y=y[0, :, 2].tolist(),overwrite=True)
            # experiment.log_curve("res Vy", x=np.arange(timestep_size).tolist(), y=y[0, :, 3].tolist(),overwrite=True)

            with experiment.test():
                # Run a validation loop at the end of each epoch.
                X_val, y_val = next(generator)
                pred_val = model(X_val, training=False)
                # Update val metrics
                val_loss_metric.update_state(y_val, pred_val)
                val_loss = val_loss_metric.result()
                experiment.log_metric("val_loss", val_loss)
                print("Epoch:", epoch, "Validation loss: %.4f" % (float(val_loss),))
                val_loss_metric.reset_states()
        # training_history  = model.fit(#x=generator,
        #                               x=X_train, y=y_train,
        #                               steps_per_epoch=1,#steps_per_epoch,
        #                               # validation_data=(X_val,y_val),
        #                               batch_size=batch_size, epochs=epochs,
        #                               verbose=1,
        #                               shuffle=True,
        #                               callbacks=[
        #                                   # ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, min_delta=1e-4, mode='min'),
        #                               #     # ModelCheckpoint(f'{base_path}/RNN_{SEED}_{fold}.hdf5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min'),
        #                               #     EarlyStopping(monitor='loss', min_delta=1e-4, patience=5, mode='min', verbose=1, baseline=None, restore_best_weights=True),
        #                                   tf.keras.callbacks.TensorBoard(
        #                                       log_dir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        #                                       histogram_freq=4,
        #                                       write_graph=True,
        #                                       # write_images=True,
        #                                       update_freq='batch',
        #                                       profile_batch=2  # ,
        #                                       # embeddings_freq=1
        #                                   )
        #                                 ]
        #                               )

    # model.load_weights(f'{base_path}/RNN_{SEED}_.hdf5')
    # fvalid = model.predict(X_val, y_val)



    experiment.end()


    #visualize gradients
    # X, y = next(generator)
    # grads_all = []
    #
    # grads_all = get_gradients(model, 1, X, y)
    # features_1D(grads_all, n_rows=2, share_xy=(1, 1))
    # grads_all = get_gradients(model, 2, X, y)
    # features_1D(grads_all, n_rows=2, share_xy=(1, 1))
    # grads_all = get_gradients(model, 3, X, y)
    # features_1D(grads_all, n_rows=2, share_xy=(1, 1))
    # plt.show()


