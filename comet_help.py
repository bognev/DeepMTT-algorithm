from tensorflow.keras import backend as K
import numpy as np

def log_histogram(experiment, gradmap, step, prefix=None):
    for k, v in gradmap.items():
        experiment.log_histogram_3d(v, name="%s/%s" % (prefix, k), step=step)


def log_weights(experiment, model, step):
    for tv in model.trainable_variables:
        experiment.log_histogram_3d(tv.numpy(), name="%s" % tv.name, step=step)

    return


def get_activations(activation_map, X, model):
    input = model.input
    outputs = [layer.output for layer in model.layers]
    functors = [K.function([input], [out]) for out in outputs]

    for func, layer in zip(functors, model.layers):
        layer_output = func([X])
        activation_map.setdefault(layer.name, 0)
        activation_map[layer.name] += np.array(layer_output)

    return activation_map


def get_gradients(gradmap, grads, model):
    for grad, param in zip(grads, model.trainable_variables):
        gradmap.setdefault(param.name, 0)
        gradmap[param.name] += grad

    return gradmap


#Data preprocessing 1-the maximum value in the entire batch data as normalization
def data_preprocessing1(data):
    weight = np.max(np.abs(data))
    results = data/weight
    return results, weight

#Each data in the batch is normalized according to the maximum value of the first value
def data_preprocessing2(data):
    weight = np.max(np.abs(data[:,0,:]), axis=1)
    weight = np.transpose(np.array([[weight]]),[2,1,0])
    results = data/weight
    return results, weight


def make_dataset():
    batch_size = 256
    generator = make_generator(batch_size, timestep_size, 1)
    trajectory, observation, est_trajectory, residual, est_trajectory_norm, median = next(generator)
    plot_distribution(est_trajectory)
    plot_distribution(trajectory)
    plot_distribution(residual)
    plot_distribution(est_trajectory_norm)


def plot_distribution(data):
    num_bins = 128

    fig10, ax10 = plt.subplots(4)
    fig10.suptitle('Hist')
    # the histogram of the data
    ax10[0].hist(data[:, :, 0].ravel(), num_bins)
    ax10[1].hist(data[:, :, 1].ravel(), num_bins)
    ax10[2].hist(data[:, :, 2].ravel(), num_bins)
    ax10[3].hist(data[:, :, 3].ravel(), num_bins)
    plt.subplots_adjust(left=0.15)


def plot_trajectory():

    batch_size = 128
    generator = make_generator(batch_size, timestep_size, 1)
    trajectory, observation, est_trajectory, residual, est_trajectory_norm, median = next(generator)

    fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})
    fig2.suptitle('Polar observations')
    for i in range(0, batch_size):
        ax2.plot(observation[i, :, 0], observation[i, :, 1])
    ax2.grid(True)

    fig5, ax5 = plt.subplots(1)
    fig5.suptitle('Dx Dy Original Trajectory/Estimated Trajectory')
    for i in range(0, batch_size):
        ax5.plot(trajectory[i, :, 0], trajectory[i, :, 1])
        ax5.plot(est_trajectory[i, :, 0], est_trajectory[i, :, 1])
    ax5.set_xlabel('Dx')
    ax5.set_ylabel('Dy')
    ax5.grid(True)

    fig6, ax6 = plt.subplots()
    fig6.suptitle('Normalized features Dx Dy ')
    for i in range(0, batch_size):
        ax6.plot(est_trajectory_norm[i, :, 0], est_trajectory_norm[i, :, 1])
    ax6.set_xlabel('Dx')
    ax6.set_ylabel('Dy')
    ax6.grid(True)

    fig2, ax2 = plt.subplots()
    fig2.suptitle('Unnormalized features Vx, Vy ')
    for i in range(0, batch_size):
        ax2.plot(est_trajectory[i, :, 2],est_trajectory[i, :, 3])
    ax2.set_xlabel('Vx')
    ax2.set_ylabel('Vy')
    ax2.grid(True)

    fig3, ax3 = plt.subplots()
    fig3.suptitle('Normalized features Vx, Vy ')
    for i in range(0, batch_size):
        ax3.plot(est_trajectory_norm[i, :, 2], est_trajectory_norm[i, :, 3])
    ax3.set_xlabel('Vx')
    ax3.set_ylabel('Vy')
    ax3.grid(True)

    fig4, ax4 = plt.subplots(4)
    fig4.suptitle('Residuals Dx, Dy, Vx,Vy')
    for i in range(0, batch_size):
        ax4[0].plot(abs(residual[i, :, 0]))
        ax4[1].plot(abs(residual[i, :, 1]))
        ax4[2].plot(abs(residual[i, :, 2]))
        ax4[3].plot(abs(residual[i, :, 3]))
        ax4[0].plot(median[:, 0])
        ax4[1].plot(median[:, 1])
        ax4[2].plot(median[:, 2])
        ax4[3].plot(median[:, 3])
    ax4[0].grid(True)
    ax4[1].grid(True)
    ax4[2].grid(True)
    ax4[3].grid(True)