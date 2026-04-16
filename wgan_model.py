from numpy import mean
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras import backend
# qcbm uses adam as an optimiser
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.constraints import Constraint
from matplotlib import pyplot as plt
from bars_and_stripes import *
# correctly encode the input in the classical model
import tensorflow as tf


# define distance between the distributions of generated and real samples
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)


# load and prepare a dataset (3x3)
def generate_real_samples(n_dimensions=3):
    dataset = get_bars_and_stripes(n_dimensions)
    # introduce the 3rd dimension (channels) to the data samples
    trainX = dataset.reshape(dataset.shape[0], 3, 3, 1)
    # scale the entries to a tanh suitable format [0 -> -1 and 1 -> 1]
    trainX = trainX * 2.0 - 1.0
    trainY = -ones((trainX.shape[0], 1))
    return trainX, trainY

# minibatch but could be removed due to the limited number of available patterns for the training
# produce sample n_samples from the dataset with replacement (bootstrapping)
# def randomise_real_samples(dataset, n_samples=14):
#     ix = randint(0, dataset.shape[0], n_samples)
#     # X- matrix of n chosen samples with indexes corresponding to ix
#     X = dataset[ix]
#     # label real samples as (-1's) (Wasserstein loss convention)
#     y = -ones((n_samples,1))
#     return X, y


# clip model - restrict weight values within a predefined boundary range [-clip_value, clip_value]
class ClipConstraint(Constraint):
    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)


# critic - extracts features and classifies the input
def critic(in_shape=(3,3,1)):
    # toDo: 1 - check different values for the randomisation processes
    init = RandomNormal(stddev=0.02)
    const = ClipConstraint(0.01)
    model = Sequential()
    # process an entire input at once with 8 (3x3) kernels due to low pattern diversity
    # toDo: test with padding="valid"
    # toDo: check other strides/filters/layers' configurations
    model.add(Input(shape=(in_shape,)))
    model.add(Conv2D(8, (3,3), strides=(1,1), padding="same", kernel_initializer=init, kernel_constraint=const))
    model.add(BatchNormalization())
    # toDo: leakyReLU
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(8, (3,3), strides=(1,1), padding='same', kernel_initializer=init, kernel_constraint=const))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    # fully connected layer with 1 output value
    model.add(Dense(1))
    # Root Mean Squared Propagation Optimiser
    # toDo: learning rate + optimisers' types
    opt = RMSprop(learning_rate=0.0025)
    # assign loss function and optimiser to the model for training
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model

# toDO final: remove batch normalisation and the second conv2d and 4 or 2 filters only


# generator - generates the output from a random input vector of size latent_dim
def generator(latent_dim):
    init = RandomNormal(stddev=0.03)
    model = Sequential()
    # n_nodes -> channels * height * width
    # (represent input as a 1D data structure - corresponding to the dimensions and channels produced by the generator)
    # in my case due to padding="same", stride 1,1 and kernel 3x3 the width and height dimensions do not get reduced but channels are intro instead
    n_nodes = 8 * 3 * 3
    # latent_dim defines the number of RANDOM input variables used to generate the output;
    # each pixel depends on all latent_dim
    model.add(Input(shape=(latent_dim,)))
    model.add(Dense(n_nodes, kernel_initializer=init))
    model.add(LeakyReLU(negative_slope=0.2))
    # restructure the input into a 3d feature map
    model.add(Reshape((3, 3, 8)))
    # introduce spatial dependencies (feature representations) across all neighbouring pixels
    # toDo: filters
    model.add(Conv2DTranspose(8, (3,3), strides=(1,1), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(negative_slope=0.2))
    # represent the output as a single feature map combining all feature maps into a single output (a generated image)
    model.add(Conv2D(1, (3,3), activation='tanh', padding='same', kernel_initializer=init))
    return model


# GAN
# generator training phase: train the generator, keep the critic fixed (freeze its weights)
def gan(g_model, c_model):
    # keep BatchNormalization layers trainable to avoid unstable training
    for layer in c_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(c_model)
    # toDo: learning rate
    opt = RMSprop(learning_rate=0.0001)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model


# generate n_samples (14) latent random vectors of dimension latent_dim
# n_samples=14 to match the batch size
def generate_latent_points(latent_dim, n_samples=14):
    # x_input - produces a long flat array with random values
    x_input = randn(latent_dim * n_samples)
    # reorganises the random values so that each data point gets latent_dim random values assigned
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input
# toDo: latent_dim to be equal to 2 or 3

#print(generate_latent_points(2))

# fake examples (generate full data samples from the vectors storing latent_dim number of variables)
def generate_fake_samples(generator, latent_dim, n_samples=14):
    x_input = generate_latent_points(latent_dim, n_samples)
    # generate fake samples from latent vectors
    X = generator.predict(x_input)
    # set all the fake samples to 1's - in contrast to real samples labeling [-1's]; (Wasserstein loss convention)
    y = ones((n_samples, 1))
    return X, y

### NOW HERE A REALLY important section S
# todo: adjust the number of the training steps to the qcbm model implementation
def train(g_model, c_model, gan_model, latent_dim, n_steps=1000, n_critic=5):
    # batches_per_epoch = int(dataset.shape[0]/n_batch)
    # n_training_steps = batches_per_epoch * n_epochs
    # half_batch = int(n_batch / 2)

    # track the history of losses for both the critic and the generator models
    c_real_history, c_fake_history, g_history = list(), list(), list()

    for i in range(n_steps):
        c_real_tmp, c_fake_tmp = list(), list()
        # critic training (half batch of real and half batch of fake samples)
        for _ in range(n_critic):
            # train on the real datapoints
            X_real, y_real = generate_real_samples()
            # update the parameters of c and keep the loss value
            c_loss_real = c_model.train_on_batch(X_real, y_real)
            c_real_tmp.append(c_loss_real)

            # train on the fake datapoints
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim)
            c_loss_fake = c_model.train_on_batch(X_fake, y_fake)
            c_fake_tmp.append(c_loss_fake)

        c_real_history.append(mean(c_real_tmp))
        c_fake_history.append(mean(c_fake_tmp))

        X_gan = generate_latent_points(latent_dim)
        # label generated samples as real to penalise or reward the generator
        # depending on how well it deluded the critic
        y_gan = -ones((len(X_gan), 1))
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        g_history.append(g_loss)

        # keep track of the training process
        print('iteration:%d/1500, critic_real_loss=%.3f, critic_fake_loss=%.3f generator_loss=%.3f' % (i+1, c_real_history[-1], c_fake_history[-1], g_loss))

    # line plots of loss
    plot_history(c_real_history, c_fake_history, g_history)


def plot_history(c_real_history, c_fake_history, g_history):
    plt.plot(c_real_history, label='critic_real_loss')
    plt.plot(c_fake_history, label='critic_fake_loss')
    plt.plot(g_history, label='generator_loss')
    plt.legend()
    plt.savefig('plot_loss.png')
    plt.close()


if __name__ == '__main__':
#     # choice of latent_dim (at least log_2(14)~14)
    latent_dim = 2
    critic = critic()
    generator = generator(latent_dim)
    gan_model = gan(generator, critic)
    train(generator, critic, gan_model, latent_dim)



