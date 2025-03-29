import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Lambda, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K
import numpy as np
import pathlib
import matplotlib.pyplot as plt
# Define input shape
input_shape = (64, 64, 3)
latent_dim = 64  # Latent space dimension

# --- Encoder using VGG16 ---
base_model = VGG16(include_top=False, input_shape=input_shape, weights='imagenet')
base_model.trainable = False  # Freeze VGG16 layers

encoder_input = Input(shape=input_shape)
x = base_model(encoder_input)
x = Flatten()(x)
z_mean = Dense(latent_dim, name="z_mean")(x)
z_log_var = Dense(latent_dim, name="z_log_var")(x)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

# Define the encoder model
encoder = Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

# --- Decoder ---
decoder_input = Input(shape=(latent_dim,))
x = Dense(8 * 8 * 256, activation='relu')(decoder_input)
x = Reshape((8, 8, 256))(x)
x = Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(x)  # 16x16
x = Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x)   # 32x32
x = Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(x)   # 64x64
x = Conv2DTranspose(3, (3, 3), strides=1, padding='same', activation='sigmoid')(x) # Keep 64x64

decoder = Model(decoder_input, x, name="decoder")

# --- VAE Model ---
vae_input = Input(shape=input_shape)
z_mean, z_log_var, z = encoder(vae_input)
vae_output = decoder(z)
vae = Model(vae_input, vae_output, name="vae")

# Loss function
reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(vae_input, vae_output))
kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
vae_loss = reconstruction_loss + kl_loss

vae.add_loss(vae_loss)
vae.compile(optimizer=tf.keras.optimizers.Adam())

# --- Load Your Dataset ---
data_dir = pathlib.Path("segmented_letters/input")

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    label_mode=None,  # No labels for VAE
    image_size=(64, 64),
    batch_size=32
)

dataset = dataset.map(lambda x: x / 255.0)  # Normalize to [0,1]

# Split dataset (80% train, 20% test)
train_size = int(0.8 * len(dataset))
train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

# --- Train the Model ---
vae.fit(train_dataset, epochs=50, validation_data=test_dataset)

# Save the trained models
vae.save("vae_model.h5")
encoder.save("encoder_model.h5")
decoder.save("decoder_model.h5")
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model

# Load trained models
decoder = load_model("decoder_model.h5")

# Function to generate a letter
def generate_letter():
    z_sample = np.random.normal(size=(1, latent_dim))  # Random latent vector
    generated_image = decoder.predict(z_sample)  # Generate an image
    return (generated_image[0] * 255).astype(np.uint8)  # Convert to displayable format

# Generate letters for "Hi how are you"
letters = "Hi how are you".replace(" ", "")  # Ignore spaces
generated_images = [generate_letter() for _ in letters]

# Resize images for uniformity
resized_images = [cv2.resize(img, (64, 64)) for img in generated_images]

# Concatenate images horizontally
sentence_image = cv2.hconcat(resized_images)



cv2.imwrite("generated_sentence.png", sentence_image)

