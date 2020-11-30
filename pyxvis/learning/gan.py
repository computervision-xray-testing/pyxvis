# gan_main.py
# GAN for images of NxN pixels (grayvalues)
# (c) D.Mery, 2019-2020
#
# Example for Training: For generating shuriken images of 32x32 pixels in 2500 epochs (and saving models and images each 250 epochs):
# python3 gan_main.py 0 shuriken_32x32.npy 2500 250
#
# For generating 1000 synthetic images of shuriken of 32x32 pixels using model learned after 1000 epochs:
# python3 gan_main.py 1 32 models/gan_model_002500.h5 1000
#
# See readme.txt for more examples


from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import h5py
import matplotlib.pyplot as plt
import sys
import numpy as np
import os.path

# Constants
GAN_PATH = '../output/GAN/'

# based on implementation of
# https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py

class DCGAN():
    def __init__(self,st):
        # load training data 
        if os.path.isfile(st):
            path_file = st
            self.X = load_train_defects(path_file)
            self.gansize = self.X.shape[1]
        else: 
            print(st)
            self.gansize = int(st)
            print(self.gansize)
        self.nmax = 3 # 3 layers in generator and discriminator

        # Input shape
        self.img_rows = self.gansize
        self.img_cols = self.gansize
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.path_file = ''

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates images
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()
        
        [p,d] = generator_size(self.gansize,self.nmax)
        
        n = len(d)

        model.add(Dense(d[0] * p * p, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((p, p, 128)))
        
        for i in range(n):
            model.add(UpSampling2D())
            model.add(Conv2D(d[i], kernel_size=3, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))
            
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        # model.add(ZeroPadding2D(padding=((0,1),(0,1)))) it was in after 64's layer

        d = 32

        model.add(Conv2D(d, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        for i in range(self.nmax):
            d = 2*d
            model.add(Conv2D(d, kernel_size=3, strides=2, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Training data
        X_train = np.expand_dims(self.X, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs+1):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            images = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_images = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(images, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_images, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)


            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_gan_examples(epoch)

            if epoch % (save_interval) == 0:
                st = '%0*d' % (6,epoch)
                self.save_synthetic_images(st,1000)
                self.save_gan_model(st)

    def save_gan_examples(self, epoch=-1):
        r, c = 5, 5
        t = r*c
        if epoch >= 0: 
            st = '%0*d' % (6,epoch)
        else:
            st = 'output'
        st_path = GAN_PATH + 'examples/gan_examples_' + st + '.png'
        print('> saving '+str(t)+ ' synthetic images (examples) in ' + st_path)

        noise = np.random.normal(0, 1, (t, self.latent_dim))
        gen_images = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_images = 0.5 * gen_images + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_images[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1

        fig.savefig(st_path)
        plt.close()


    def save_synthetic_images(self, st, N):
        noise = np.random.normal(0, 1, (N, self.latent_dim))
        gen_images = self.generator.predict(noise)
        # print('> computing '+str(N) + ' GAN synthetic images...')

        # Rescale images 0 - 1
        gen_images = 0.5 * gen_images + 0.5
        save_images_gan(N,gen_images,st)


    def save_gan_model(self,st):
        st_path = GAN_PATH + 'models/gan_model_'+st+'.h5'
        print('> saving model in ' + st_path + ' ...')
        save_model(self.combined,st_path)

    def load_gan_model(self,st_path):
        # st_path = 'models/gan_model_'+st+'.h5'
        print('loading ' + st_path + ' ...')
        load_model(self.combined,st_path)

def save_images_gan(N,gen_images,st):
        st_path = GAN_PATH + 'synthetic/gan_synthetic_'+st
        print('> saving '+str(N)+' GAN synthetic images in '+st_path+'.npy')
        np.save(st_path,gen_images)
        return

def load_train_defects(st_file):
        print('loading training defect pacthes '+st_file+'...')
        x_train = np.load(st_file)
        return x_train

# from https://github.com/keras-team/keras/issues/10608

def save_model(model, filename):
    f = h5py.File(filename,'w')
    for symbolic_weights, weights in zip(model.weights, model.get_weights()):
        name = symbolic_weights.name
        f.create_dataset(name, data=weights)
    f.flush()
    f.close()
    
def load_model(model, filename):
    f = h5py.File(filename, 'r')
    weights = []
    for symbolic_weights in model.weights:
        name = symbolic_weights.name
        dataset = f[name]
        weights.append( dataset[:] )
    model.set_weights(weights)
    f.close()

def generator_size(p,nmax):

    n = 0
    ok = 0

    while not(ok):
        t = p/2
        if 2*np.fix(t)==p:
            n = n+1
            p = t
            if n == nmax:
                ok = 1
        else:
            ok = 1
        
    d = np.zeros((n,), dtype=int)
    g = 128
    for i in range(n):
        d[i] = g
        g = g/2

    p = int(p)
        # d = int(d)
        
    print(p)

    print(d)

    print(n)

    return p,d


