from maxout import max_out as MO     #maxout activation function
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K
import tensorflow.keras.layers as L
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import tensorflow as tf

#noisy activation function
def noisy_af(Xt):
    p = 1
    c = 1
    h = tf.nn.relu(0.5*Xt+0.5)-tf.nn.relu(0.5*Xt-0.5) - 0.5
    y = h + c*tf.square(K.sigmoid(p*(h-Xt))-0.5) * tf.random.normal(tf.shape(Xt))
    return y

#piecewise activation function
def piecewise(Xt):
    y = tf.nn.relu(Xt+1)-tf.nn.relu(Xt-1)-1
    return y

get_custom_objects().update({'noisy_af': noisy_af})
get_custom_objects().update({'piecewise': piecewise})




class FIRFilter(tf.keras.layers.Layer):
    def __init__(self, input_shape, **kwargs):
        super(FIRFilter, self).__init__(**kwargs)
        self.initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)
        self.fir_size, self.input_size, self.batch_size, self.timestep_size = input_shape

    def build(self, input_shape):

        self.fir_w = self.add_weight(name='fir_w',
                                    shape=(self.fir_size, self.input_size),
                                    initializer=tf.keras.initializers.Constant(1.0 / self.fir_size),
                                    trainable = True)
        self.x_add = tf.constant(name='x_add',
                                 value=tf.zeros((self.batch_size, self.fir_size - 1, self.input_size)),
                                 dtype=tf.float32,
                                 shape=(self.batch_size, self.fir_size-1, self.input_size))

    def call(self, inputs, **kwargs):
        inputs = tf.concat([self.x_add, inputs], 1)  # Fill in zeros for the sequence to be filtered with insufficient length before
        inputs = tf.expand_dims(inputs, 2)  # Expand one dimension, store the data to be filtered
        y = []
        for i in range(self.fir_size):
            y.append(inputs[:, i:i+self.timestep_size, :, :])
        z = tf.concat(y, 2)
        #F.7 https: // www.dsprelated.com / freebooks / filters / Matrix_Filter_Representations.html
        z = self.fir_w * z
        return tf.reduce_sum(z, axis=2)

    # Implement get_config to enable serialization. This is optional.
    def get_config(self):
        base_config = super(FIRFilter, self).get_config()
        config = {"initializer": tf.keras.initializers.serialize(self.initializer)}
        return dict(list(base_config.items()) + list(config.items()))

#------------------------------------------------------------------
#Input and output definition

class Maxout(tf.keras.layers.Layer):
    def __init__(self, maxout_size, timestep_size,  hidden_size, output_size, lambda1, **kwargs):
        super(Maxout, self).__init__(**kwargs)
        self.output_size, self.timestep_size, self.hidden_size = output_size, timestep_size, hidden_size
        self.maxout_size, self.lambda1 = maxout_size, lambda1
        self.initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)#'glorot_uniform'#

    def build(self, input_shape):

        self.dense_in = L.Dense(units=self.hidden_size,
                                kernel_initializer=tf.keras.initializers.glorot_normal(),
                                bias_initializer=tf.keras.initializers.Constant(0.01),
                                kernel_regularizer = tf.keras.regularizers.l2(l2=self.lambda1))
        self.maxout = MaxoutLayer(num_units=self.maxout_size, axis=2)
        self.dense_out = L.Dense(units=self.output_size,
                                kernel_initializer=tf.keras.initializers.glorot_normal(),
                                bias_initializer=tf.keras.initializers.Constant(0.01),
                                kernel_regularizer = tf.keras.regularizers.l2(l2=self.lambda1))


    def call(self, inputs, **kwargs):
        # tensroflow's three-dimensional tensor is multiplied, the first dimension is ignored, and the next two dimensions are multiplied, so here put the timestep_size in the first dimension first
        maxout_in = self.dense_in(tf.transpose(inputs, [1, 0, 2]))
        maxout_out = self.maxout(tf.transpose(maxout_in, [1, 0, 2]))  # MAXOUT
        y_pred = self.dense_out(tf.transpose(maxout_out, [1, 0, 2]))
        # maxout_in = self.dense_in(inputs)
        # maxout_out = self.maxout(maxout_in)  # MAXOUT
        # y_pred = self.dense_out(maxout_out)

        return tf.transpose(y_pred, [1, 0, 2])

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(Maxout, self).get_config()
        config = {"initializer": tf.keras.initializers.serialize(self.initializer)}
        return dict(list(base_config.items()) + list(config.items()))


from tensorflow_addons.layers import Maxout as MaxoutLayer

#Build lstm network, separate multiple layers
def lstm_model(batch_size, timestep_size, input_size, keep_prob, fir_size, hidden_size_1, hidden_size_2, hidden_size_3):
    activation = "tanh"#"noisy_af"
    X = L.Input(name='X', shape=(timestep_size, input_size), dtype=tf.float32, batch_size=batch_size)
    # FIR
    x_filtered = FIRFilter(input_shape=(fir_size, input_size, batch_size, timestep_size))(X)
    # forward:
    lstm_cell_1_fw = L.LSTMCell(units=hidden_size_1, unit_forget_bias=True, dropout=keep_prob, recurrent_dropout=keep_prob, activation=activation, kernel_initializer=tf.keras.initializers.glorot_normal(), recurrent_initializer=tf.keras.initializers.glorot_normal())
    outputs_l1_fw = L.RNN(lstm_cell_1_fw, return_sequences=True, time_major=False)
    init_outputs_l1_fw = outputs_l1_fw.get_initial_state(tf.random.normal([batch_size, timestep_size, input_size]))
    # backward:
    lstm_cell_1_bw = L.LSTMCell(units=hidden_size_1, unit_forget_bias=True, dropout=keep_prob, recurrent_dropout=keep_prob, activation=activation, kernel_initializer=tf.keras.initializers.glorot_normal(), recurrent_initializer=tf.keras.initializers.glorot_normal())
    outputs_l1_bw = L.RNN(lstm_cell_1_bw, go_backwards=True, return_sequences=True, time_major=False)
    init_outputs_l1_bw = outputs_l1_fw.get_initial_state(tf.random.normal([batch_size, timestep_size, input_size]))
    outputs_l1 = L.Bidirectional(outputs_l1_fw, backward_layer=outputs_l1_bw, merge_mode="concat", name="Bidirectional_0")(inputs=x_filtered, initial_state=init_outputs_l1_fw+init_outputs_l1_bw)

    #forward:
    lstm_cell_2_fw = L.LSTMCell(units=hidden_size_2, unit_forget_bias=True, dropout=keep_prob, recurrent_dropout=keep_prob, activation=activation, kernel_initializer=tf.keras.initializers.glorot_normal(), recurrent_initializer=tf.keras.initializers.glorot_normal())
    outputs_l2_fw = L.RNN(lstm_cell_2_fw, return_sequences=True,time_major=False)
    init_outputs_l2_fw = outputs_l2_fw.get_initial_state(tf.random.normal([batch_size, timestep_size, hidden_size_2]))
    #backward:
    lstm_cell_2_bw = L.LSTMCell(units=hidden_size_2, unit_forget_bias=True, dropout=keep_prob, recurrent_dropout=keep_prob, activation=activation, kernel_initializer=tf.keras.initializers.glorot_normal(), recurrent_initializer=tf.keras.initializers.glorot_normal())
    outputs_l2_bw = L.RNN(lstm_cell_2_bw, return_sequences=True, go_backwards=True, time_major=False)
    init_outputs_l2_bw = outputs_l2_fw.get_initial_state(tf.random.normal([batch_size, timestep_size, hidden_size_2]))
    outputs_l2 = L.Bidirectional(outputs_l2_fw, backward_layer=outputs_l2_bw, merge_mode="concat", name="Bidirectional_1")(inputs=outputs_l1, initial_state=init_outputs_l2_fw+init_outputs_l2_bw)

    #forward:
    lstm_cell_3_fw = L.LSTMCell(units=hidden_size_3, unit_forget_bias=True, kernel_initializer=tf.keras.initializers.glorot_normal(), recurrent_initializer=tf.keras.initializers.glorot_normal())
    outputs_l3_fw = L.RNN(lstm_cell_3_fw, return_sequences=True, time_major=False)
    init_outputs_l3_fw = outputs_l3_fw.get_initial_state(tf.random.normal([batch_size, timestep_size, hidden_size_3]))
    #backward:
    lstm_cell_3_bw = L.LSTMCell(units=hidden_size_3, unit_forget_bias=True, kernel_initializer=tf.keras.initializers.glorot_normal(), recurrent_initializer=tf.keras.initializers.glorot_normal())
    outputs_l3_bw = L.RNN(lstm_cell_3_bw, go_backwards=True, return_sequences=True, time_major=False)
    init_outputs_l3_bw = outputs_l3_bw.get_initial_state(tf.random.normal([batch_size, timestep_size, hidden_size_3]))
    outputs_l3 = L.Bidirectional(outputs_l3_fw, backward_layer=outputs_l3_bw, merge_mode="concat", name="Bidirectional_2")(inputs=outputs_l2, initial_state=init_outputs_l3_fw+init_outputs_l3_bw)

    residual_pred = Maxout(input_shape=outputs_l3.shape)(outputs_l3)

    model = tf.keras.Model(inputs=[X], outputs=[residual_pred])
    # model.compile(optimizer=tf.optimizers.RMSprop(lr=0.01), loss=root_mean_squared_error, metrics=['mae'])
    return model

def create_fmodel(batch_size, timestep_size, input_size, keep_prob):
    input = L.Input(shape=(timestep_size,input_size), batch_size=batch_size)
    x = L.BatchNormalization()(input)
    x = L.Dense(64, activation='relu')(x)
    # x = L.BatchNormalization()(x)
    forward_layer = L.LSTM(256, dropout=0.3, recurrent_dropout=0, activation='tanh', return_sequences=True, stateful=False)
    backward_layer = L.LSTM(256, dropout=0.3, recurrent_dropout=0, activation='tanh', return_sequences=True, go_backwards=True, stateful=False)
    x = L.Bidirectional(layer=forward_layer, backward_layer=backward_layer)(x)
    x = L.Bidirectional(L.LSTM(512, dropout=0.3, return_sequences=True, activation='tanh'))(x)
    # x = L.BatchNormalization()(x)
    x = L.Dense(64, activation='tanh')(x)
    # x = L.BatchNormalization()(x)
    output = L.Dense(4, activation=None, name='floor')(x)
    model = tf.keras.Model(inputs = [input], outputs=[output])
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=init_lr, decay=init_lr / epochs), loss=loss_fn, metrics=['mae'])
    return model


from temporal_attention import *

def create_gmodel(batch_size, timestep_size, input_size, keep_prob):
    input = L.Input(shape=(timestep_size,input_size), batch_size=batch_size)
    x = L.BatchNormalization()(input)
    x = L.Dense(64, activation='relu')(x)
    # x = L.BatchNormalization()(x)
    forward_layer = L.LSTM(256, dropout=0.3, recurrent_dropout=0, activation='tanh', return_sequences=True, stateful=False)
    backward_layer = L.LSTM(256, dropout=0.3, recurrent_dropout=0, activation='tanh', return_sequences=True, go_backwards=True, stateful=False)
    x = L.Bidirectional(layer=forward_layer, backward_layer=backward_layer)(x)
    x = L.Bidirectional(L.LSTM(512, dropout=0.3, return_sequences=True, activation='tanh'))(x)
    # x = L.BatchNormalization()(x)
    attention = TemporalAttention(timestep_size, return_attention=False)(x)
    # x = L.BatchNormalization()(x)
    # x = L.Dense(1024)(attention)
    output = L.Dense(4, activation=None, name='floor')(tf.reshape(attention,(batch_size, timestep_size,16 )))
    model = tf.keras.Model(inputs = [input], outputs=[output])
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=init_lr, decay=init_lr / epochs), loss=loss_fn, metrics=['mae'])
    return model
