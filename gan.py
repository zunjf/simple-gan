#%%
import numpy as np 
from tensorflow import keras

from keras import Sequential
from keras.layers import Dense, BatchNormalization, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten
from keras.datasets import mnist
from keras.optimizers import Adam

#%%
# Declare Generator Network
def generator(width, height, channels):
    model = Sequential()
    model.add(Dense(256, input_shape=(100,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(width*height*channels, activation='tanh'))
    model.add(Reshape((width, height, channels)))

    return model

#%%
def discriminator(width, height, channels):
    
    model = Sequential()
    model.add(Flatten(input_shape=(width, height, channels)))
    model.add(Dense((width*height*channels), input_shape=(width,height,channels)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(int((width*height*channels)/2)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    return model

def stacked_generator_discriminator(gen, dis):
    dis.trainable = False 

    model = Sequential()
    model.add(gen)
    model.add(dis)

    return model

def train():
    pass

def plot_images():
    pass

#%%
# Main function
if __name__ == '__main__':

    # Load data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = np.expand_dims(train_images, axis=3)
    
    # Parameters
    img_width = train_images.shape[1]
    img_height = train_images.shape[2]
    img_channel = train_images.shape[3]
    shape = (img_width, img_height, img_channel)
    epoch = 20
    batch = 32
    save_interval = 100

    # Generator
    G = generator(img_width, img_height, img_channel)
    G.compile(optimizer=Adam(lr=0.0002, beta_1=0.5, decay=8e-8), 
              loss='binary_crossentropy')

    # Discriminator
    D = discriminator(img_width, img_height, img_channel)
    D.compile(optimizer=Adam(lr=0.0002, decay=8e-9), 
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
    # Stacked Generator and Discriminator
    G_D = stacked_generator_discriminator(G, D)
    G_D.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, decay=8e-9))

    for cnt in range(epoch):
        ### Train Discriminator
        # Image selection for training
        random_idx = np.random.randint(0, len(train_images) - batch)
        legit_img = train_images[random_idx:int(random_idx+batch/2)].reshape(int(batch/2), img_width, img_height, img_channel)

        # Generate noise
        gen_noise = np.random.normal(0, 1, (int(batch/2), 100))
        print(gen_noise.shape)
        syntetic_img = G.predict(gen_noise)
        print(syntetic_img.shape)

        # Combine synthetic data and original data
        # Label legit data with 1 and synthetic with 0
        x_combined_batch = np.concatenate((legit_img, syntetic_img))
        print(x_combined_batch.shape)
        y_combined_batch = np.concatenate((np.ones((int(batch/2), 1)), np.zeros((int(batch/2), 1))))
        print(y_combined_batch)

        # Runs a single gradient update on a single batch of data
        batch_loss = D.train_on_batch(x_combined_batch, y_combined_batch)

        print(batch_loss)

        ### Train Generator
        

    # https://medium.com/@mattiaspinelli/simple-generative-adversarial-network-gans-with-keras-1fe578e44a87