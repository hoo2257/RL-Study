import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


def gpu_setting(memory):
    ################### Limit GPU Memory ###################
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("########################################")
    print('{} GPU(s) is(are) available'.format(len(gpus)))
    print("########################################")

    # set the only one GPU and memort limit
    memory_limit = memory * 1024

    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
            print("Use only one GPU{} limited {}MB memory".format(gpus[0], memory_limit))
        except RuntimeError as e:
            print(e)

    else:
        print('GPU is not available')