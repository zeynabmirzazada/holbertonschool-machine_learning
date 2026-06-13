#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


class Simple_GAN(keras.Model):
    def __init__(
        self,
        generator,
        discriminator,
        latent_generator,
        real_examples,
        batch_size=200,
        disc_iter=2,
        learning_rate=0.005,
    ):
        super().__init__()  # run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.5  # standard value, but can be changed if necessary
        self.beta_2 = 0.9  # standard value, but can be changed if necessary

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: tf.math.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(
                optimizer=generator.optimizer, loss=generator.loss)

        # define the discriminator loss and optimizer:
        self.discriminator.loss = (
                lambda x, y: tf.math.reduce_mean(y) -
                tf.math.reduce_mean(x))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(
            optimizer=discriminator.optimizer, loss=discriminator.loss
        )

    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # overloading train_step()
    def train_step(self, useless_argument):
        for _ in range(self.disc_iter):
            with tf.GradientTape() as g:
                g.watch(self.discriminator.weights)
                x = self.discriminator(self.get_real_sample())
                y = self.discriminator(self.get_fake_sample())
                discr_loss = self.discriminator.loss(y, x)
            a = g.gradient(discr_loss, self.discriminator.weights)
            self.discriminator.optimizer.apply_gradients(
                zip(a, self.discriminator.weights)
            )
            tf.clip_by_value(self.discriminator.weights, clip_value_min=-1,
                    clip_value_max=1)

        with tf.GradientTape() as h:
            h.watch(self.generator.weights)
            x = self.discriminator(self.get_fake_sample(training=True))
            gen_loss = self.generator.loss(x)
        b = h.gradient(gen_loss, self.generator.weights)
        self.generator.optimizer.apply_gradients(
                zip(b, self.generator.weights))
        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
