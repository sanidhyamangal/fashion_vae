import tensorflow as tf  # for deep learning
import numpy as np  # for matrix maths
import pandas as pd  # for loading data frames
import time
from sklearn.model_selection import train_test_split

# load dataset
data = pd.read_csv("fashion-mnist_train.csv")

# create dataset for the models
pixels = data.iloc[:, 1:].values / 255.

# reshape data to make compatible with network
pixels = np.reshape(pixels, [-1, 28, 28, 1])

train_set, test_set = train_test_split(pixels, random_state=0, test_size=0.3)

# binarization of pixels
train_set[train_set >= 0.5] = 1.
train_set[train_set < 0.5] = 0.
test_set[test_set >= .5] = 1.
train_set[train_set < .5] = 0.

# convert data into tf tensors
train_set = tf.convert_to_tensor(train_set, dtype=tf.float32)
test_set = tf.convert_to_tensor(test_set, dtype=tf.float32)

# create a dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_set).shuffle(
    42000).batch(100)
test_dataset = tf.data.Dataset.from_tensor_slices(test_set).shuffle(
    18000).batch(100)


# create a class for
class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(filters=32,
                                   kernel_size=3,
                                   strides=(2, 2),
                                   activation="relu"),
            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=3,
                                   strides=(2, 2),
                                   activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim + latent_dim)
        ])

        # generator part
        self.generative_net = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim, )),
            tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(filters=32,
                                            kernel_size=3,
                                            strides=2,
                                            padding="same",
                                            activation="relu"),
            tf.keras.layers.Conv2DTranspose(filters=64,
                                            kernel_size=3,
                                            strides=2,
                                            padding="same",
                                            activation="relu"),
            tf.keras.layers.Conv2DTranspose(filters=1,
                                            kernel_size=3,
                                            strides=1,
                                            padding="same")
        ])

    # a sample function
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x),
                                num_or_size_splits=2,
                                axis=1)
        return mean, logvar

    def reparamaterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


# define loss function
optimizer = tf.keras.optimizers.Adam(1e-4)


# log normal function
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean)**2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


# function to compute loss
@tf.function
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparamaterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit,
                                                        labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqx_z = log_normal_pdf(z, mean, logvar)

    return -tf.reduce_mean(logpx_z + logpz + logqx_z)


# apply gradient function
@tf.function
def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


epochs = 100
latent_dim = 50
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)


for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    compute_apply_gradients(model, train_x, optimizer)
  end_time = time.time()

  if epoch % 1 == 0:
    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
      loss(compute_loss(model, test_x))
    elbo = -loss.result()
    print('Epoch: {}, Test set ELBO: {}, '
          'time elapse for current epoch {}'.format(epoch,
                                                    elbo,
                                                    end_time - start_time))