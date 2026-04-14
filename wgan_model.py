from numpy import mean
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras import backend
from keras.optimizers import RMSprop
from keras.models import Sequential
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
import tensorflow as tf


# define distance between the distributions of generated and real samples
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)


# load and prepare a dataset (3x3)
def load_real_samples(n_dimensions=3):
    dataset = get_bars_and_stripes(n_dimensions)
    # introduce the 3rd dimension (channels) to the data samples
    dataset = dataset.reshape(dataset.shape[0], 3, 3, 1)
    # rewrite 0's into (-1's); a suitable format for the tanh decision function
    dataset = dataset * 2.0 - 1.0
    return dataset


# create a random mini-batch with labels
def select_real_samples(dataset, n_samples):
    # bootstrapping - draw n_samples with replacement
    ix = randint(0, dataset.shape[0], n_samples)
    # X- matrix of n chosen samples with indexes corresponding to ix
    X = dataset[ix]
    # label real samples as (-1's) (Wasserstein loss convention)
    y = -ones((n_samples,1))
    return X, y


# clip model - restrict weight values within a predefined boundary range [-clip_value, clip_value]
class ClipConstraint(Constraint):
    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)


# critic - extracts features and classifies the input
def critic(in_shape=(3,3,1)):
    init = RandomNormal(stddev=0.02)
    const = ClipConstraint(0.01)
    model = Sequential()
    # process an entire input at once with 8 (3x3) kernels due to low pattern diversity
    # toDo: test with padding="valid"
    model.add(Conv2D(8, (3,3), strides=(1,1), padding="same", kernel_initializer=init, kernel_constraint=const, input_shape=in_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(8, (3,3), strides=(1,1), padding='same', kernel_initializer=init, kernel_constraint=const))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    # fully connected layer with 1 output value
    model.add(Dense(1))
    # Root Mean Squared Propagation Optimiser
    opt = RMSprop(learning_rate=0.00005)
    # assign loss function and optimiser to the model for training
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model


# generator - generates the output from a random input vector of size latent_dim
def generator(latent_dim):
    init = RandomNormal(stddev=0.03)
    model = Sequential()
    # n_nodes -> channels * height * width
    # (represent input as a 1D data structure)
    n_nodes = 8 * 3 * 3
    # latent_dim defines the number of RANDOM input variables used to generate the output;
    # each pixel depends on all latent_dim
    model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    # restructure the input into a 3d feature map
    model.add(Reshape((3, 3, 8)))
    # introduce spatial dependencies (feature representations) across all neighbouring pixels
    model.add(Conv2DTranspose(8, (3,3), strides=(1,1), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # represent the output as a single feature map combining all feature maps into a single output (a generated image)
    model.add(Conv2D(1, (3,3), activation='tanh', padding='same', kernel_initializer=init))
    return model


# GAN
# generator training phase: train the generator, keep the critic fixed (freeze its weights)
def gan(generator, critic):
    # keep BatchNormalization layers trainable to avoid unstable training
    for layer in critic.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(critic)
    opt = RMSprop(learning_rate=0.00005)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model


# generate n latent random vectors of dimension latent_dim
def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim, n_samples)
    # reshape the matrix so that each row is a separate latent vector
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# fake examples
def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    # generate fake samples from latent vectors
    X = generator.predict(x_input)
    # set all the fake samples to 1's - in contrast to real samples labeling [-1's]; (Wasserstein loss convention)
    y = ones((n_samples, 1))
    return X, y


def train(g_model, c_model, gan_model, dataset, latent_dim, n_epochs=500, n_batch=4, n_critic=5):
    batches_per_epoch = int(dataset.shape[0]/n_batch)
    n_training_steps = batches_per_epoch * n_epochs
    half_batch = int(n_batch / 2)

    # track the history of losses for both the critic and the generator models
    c_real_history, c_fake_history, g_history = list(), list(), list()

    for i in range(n_training_steps):
        c_real_tmp, c_fake_tmp = list(), list()
        # critic training (half batch of real and half batch of fake samples)
        for _ in range(n_critic):
            # train on the real datapoints
            X_real, y_real = select_real_samples(dataset, half_batch)
            # update the parameters of c and keep the loss value
            c_loss_real = c_model.train_on_batch(X_real, y_real)
            c_real_tmp.append(c_loss_real)

            # train on the fake datapoints
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            c_loss_fake = c_model.train_on_batch(X_fake, y_fake)
            c_fake_tmp.append(c_loss_fake)

        c_real_history.append(mean(c_real_tmp))
        c_fake_history.append(mean(c_fake_tmp))

        X_gan = generate_latent_points(latent_dim, n_batch)
        # label generated samples as real to penalise or reward the generator
        # depending on how well it deluded the critic
        y_gan = -ones((n_batch, 1))
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
    # choice of latent_dim (at least log_2(14)~14)
    latent_dim = 8
    critic = critic()
    generator = generator(latent_dim)
    gan_model = gan(generator, critic)
    dataset = load_real_samples(3)
    train(generator, critic, gan_model, dataset, latent_dim)



