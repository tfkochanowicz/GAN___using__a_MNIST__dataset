import keras
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU, Dense, Dropout, Flatten, Conv2D, Reshape, UpSampling2D, Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist

# I am just going to use the training set of images as the data
(x_train, _), (_, _) = mnist.load_data(path="mnist.npz")
x_train = x_train.reshape(60000, 28, 28, 1)
x_train = x_train.astype('float32')/255
# print('Train', x_train.shape)

inputs = keras.Input(shape=(28, 28, 1))

#Discrimantor model below
x = Conv2D(32, (5, 5), strides=(2, 2), padding='same')(inputs)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.4)(x)

x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.4)(x)

x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.4)(x)

x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.4)(x)

x = Flatten()(x)
outputs = Dense(1, activation='sigmoid')(x)
model_disc = keras.Model(inputs=inputs, outputs=outputs, name="Discriminator")
# model_disc.summary()
# after some research i figured out that the optimizer is best set to these parameters
# for both the discriminator and the GAN model
model_disc.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])


# dim number for generator

model_gen = Sequential()
dim = 100
model_gen.add(Dense(units=7 * 7 * 192, input_dim=dim))
model_gen.add(BatchNormalization(scale='relu'))
model_gen.add(Reshape((7, 7, 192)))
model_gen.add(Dropout(0.4))
model_gen.add(UpSampling2D(2))
model_gen.add(Conv2DTranspose(96, (5, 5), strides=1, padding='same'))
model_gen.add(BatchNormalization(scale='relu'))
model_gen.add(UpSampling2D(2))
model_gen.add(Conv2DTranspose(48, (5, 5), strides=1, padding='same'))
model_gen.add(BatchNormalization(scale='relu'))
model_gen.add(Conv2DTranspose(24, (5, 5), strides=1, padding='same'))
model_gen.add(BatchNormalization(scale='relu'))
model_gen.add(Conv2DTranspose(1, (5, 5), strides=1, padding='same', activation='sigmoid'))
model_gen.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
model_disc.trainable = False

model_GAN = Sequential()

model_GAN.add(model_gen)
model_GAN.add(model_disc)


model_GAN.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')

# model_disc.summary()
# model_gen.summary()
# model_GAN.summary()


batch_size = 64
epoch_size = 80
half_batch = batch_size/2
GAN_losses = []
disc_losses = []

# GAN_losses_epoch = []
# disc_losses_epoch = []
for e in range(1, epoch_size+1):
    print("Epoch: ", e)
    for b in range(batch_size):
        noise = np.random.normal(0, 1, size=(batch_size, 100))
        images_real = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
        images_fake = model_gen.predict(noise)

        images_combined = np.concatenate([images_real, images_fake])
        y_labels = np.zeros(batch_size*2)
        y_labels[:batch_size] = 0.9

        model_disc.trainable = True
        disc_loss = model_disc.train_on_batch(images_combined, y_labels)
        noise = np.random.normal(0, 1, size=[batch_size, 100])

        y_labels2 = np.ones(batch_size)
        model_disc.trainable = False
        GAN_loss = model_GAN.train_on_batch(noise, y_labels2)

        # disc_losses.append(disc_loss)
        # GAN_losses.append(gan_loss)

    disc_losses.append(disc_loss)
    GAN_losses.append(GAN_loss)

    if e == 1 or e % 10 == 0:
        noise = np.random.normal(0, 1, size=[16, 100])
        generated_images = model_gen.predict(noise)

        plt.figure(figsize=(18, 18))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(generated_images[i], cmap='gray')
        print("Epoch: ", e)
        plt.axis('off')
        plt.show()





plt.plot(disc_losses, label='Discriminator loss')
plt.plot(GAN_losses, label='Generative Adversarial loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

