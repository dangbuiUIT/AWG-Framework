from keras.layers import Input, Dense, Reshape, Flatten,  Dropout, Embedding, LeakyReLU, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, LSTM, TimeDistributed
from keras.optimizers import Adam, RMSprop
from keras.models import Model
from keras.utils import plot_model
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D
from keras.models import load_model

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import pickle
import os
import matplotlib.pyplot as plt
from keras.constraints import Constraint
from keras import backend
from keras.models import Sequential
from matplotlib import pyplot
from numpy import mean
from functools import partial

import tensorflow
import keras.backend as K
import seaborn as sns
import sys
import util

datasetr = util.load_real_samples('path_to_dataset.csv')
# gen dataset dga base CHARBOT and data_real
datasetf = util.load_fake_sample(datasetr)

# load black-box detector
detector = load_model('path_to_model')

#define basic infomation
domain_len = 25
domain_shape = (domain_len, dictionary_size)
BATCH_SIZE = 64

class WGAN():
    def __init__(self):
        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        self.img_rows = domain_shape[0]
        self.img_cols = domain_shape[1]
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 600
        optimizer = RMSprop(learning_rate=0.0001)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=RMSprop(learning_rate=0.0001),
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z, zz = self.generator.input
        gen_out = self.generator.output
        gan_out = self.critic(gen_out)

        # # For the combined model we will only train the generator
        self.critic.trainable = False

        # # The critic takes generated images as input and determines validity
        # valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model([z, zz], gan_out)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        image_input = Input(shape=(25, 40, 1))

        noise_input = Input(shape=(self.latent_dim,))

        x1 = Conv2D(128, kernel_size=4, padding="same")(image_input)
        x1 = BatchNormalization(momentum=0.8)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = Conv2D(64, kernel_size=4, padding="same")(x1)
        x1 = BatchNormalization(momentum=0.8)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = Conv2D(32, kernel_size=4, padding="same")(x1)
        x1 = BatchNormalization(momentum=0.8)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)

        x2 = Dense(128 * 25 * 40, activation="relu")(noise_input)
        x2 = Reshape((25, 40, 128))(x2)
        x2 = Conv2D(64, kernel_size=5, padding="same")(x2)
        x2 = BatchNormalization(momentum=0.8)(x2)
        x2 = LeakyReLU(alpha=0.2)(x2)
        x2 = Conv2D(32, kernel_size=5, padding="same")(x2)
        x2 = BatchNormalization(momentum=0.8)(x2)
        x2 = LeakyReLU(alpha=0.2)(x2)

        combined = Concatenate()([x1, x2])

        output = Conv2D(self.channels, kernel_size=4, padding="same")(combined)
        output = Activation("tanh")(output)

        generator = Model(inputs=[image_input, noise_input], outputs=output)
        generator.summary()
        return generator

    def build_critic(self):

        input_shape = (25, 40, 1)
        input_layer = Input(shape=input_shape)

        x = Conv2D(64, kernel_size=5, strides=2, padding="same")(input_layer)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, kernel_size=5, strides=2, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, kernel_size=5, strides=2, padding="same")(x)
        x = ZeroPadding2D(padding=((0,1),(0,1)))(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, kernel_size=5, strides=2, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        output = Dense(1)(x)

        discriminator = Model(inputs=input_layer, outputs=output)
        discriminator.summary()
        return discriminator

    def get_data_train (self, batch_size, latent_dim):
        # Sample noise as generator input
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        # Generate a batch of new images
        im = generate_fake_samples(datasetf, batch_size)
        base_imgs = []
        for i in im:
          temp = array2domain(i)
          temp, mv = domain2array(temp)
          temp = temp.reshape((25,40,1))
          base_imgs.append(temp)
        base_imgs = np.asarray(base_imgs)
        gen_imgs = self.generator.predict([base_imgs,noise])
        #
        imgs = generate_real_samples(datasetr, batch_size)
        imgs = np.asarray(imgs)
        y_real = np.ones((batch_size, 1))
        y_fake = -np.ones((batch_size, 1))

        return imgs, gen_imgs, y_real, y_fake

    def train(self, epochs, batch_size=128, sample_interval=10):

        d_loss_his = []
        g_loss_his = []
        for epoch in range(epochs):
            d_loss = 0
            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                imgs, gen_imgs, valid, fake = self.get_data_train(batch_size, self.latent_dim)
                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                temp = 0.5 * np.add(d_loss_fake, d_loss_real)
                d_loss += temp[0]

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)
            d_loss_his.append(d_loss/5)


            # ---------------------
            #  Train Generator
            # ---------------------
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            base_imgs = generate_fake_samples(datasetf, batch_size)
            base_imgs = np.asarray(base_imgs)
            valid = np.ones((batch_size, 1))
            g_loss = self.combined.train_on_batch([base_imgs,noise], valid)
            g_loss_his.append(g_loss[0])

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - (d_loss/5), 1 - g_loss[0]))

            # If at save interval => save generated image samples
            if (epoch+1) % sample_interval == 0:
              self.sample_images(epoch)

        plot_history(d_loss_his, g_loss_his)



    def sample_images(self, epoch):
        noise = np.random.normal(0, 1, (3, self.latent_dim))
        base_imgs = generate_fake_samples(datasetf, 3)
        base_imgs = np.asarray(base_imgs)
        gen_imgs = self.generator.predict([base_imgs,noise])
        for i in range(3):
          t = array2domain(gen_imgs[i])
          print(t)

# initial WIGAN and training
wgan = WGAN()
wgan.train(epochs=10000, batch_size=32, sample_interval=5)
# save Generator
wgan.generator.save('path_to_save')