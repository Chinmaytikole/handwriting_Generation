import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, Dropout, BatchNormalization

# Define image size and batch size
img_size = (64, 64)
batch_size = 32

# Load images from the segmented_letters directory
datagen = ImageDataGenerator(rescale=1.0/255.0)

train_data = datagen.flow_from_directory(
    "segmented_letters/",
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode=None  # Since it's unsupervised
)

# Define the model
model = Sequential([
    Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(64, 64, 1)),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(64*64, activation='sigmoid'),
    Reshape((64, 64))
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()

model.fit(train_data, train_data, epochs=10)


