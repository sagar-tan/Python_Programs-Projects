import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np

# Base paths
base_path = r'C:\Users\tanwa\Python\Sarthak_Project\Vehicles'
categories = ['Four Wheelers', 'Three Wheelers', 'Two Wheelers']

# Image settings
image_size = (64, 64)  # Resize all images to this size
batch_size = 32

# Helper function to load data
def load_data(category, subset):
    path = os.path.join(base_path, category, subset)
    datagen = ImageDataGenerator(rescale=1./255)
    data = datagen.flow_from_directory(
        path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    return data

# Load datasets
train_data = [load_data(category, 'train') for category in categories]
val_data = [load_data(category, 'val') for category in categories]
test_data = [load_data(category, 'test') for category in categories]

# Combine generators into a single generator
def combine_generators(generators):
    for generator in generators:
        for data in generator:
            yield data

# Create the model
cnn = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = cnn.fit(
    combine_generators(train_data),
    validation_data=combine_generators(val_data),
    epochs=10,
    steps_per_epoch=sum(len(generator) for generator in train_data),
    validation_steps=sum(len(generator) for generator in val_data)
)

# Evaluate the model
test_steps = sum(len(generator) for generator in test_data)
test_loss, test_accuracy = cnn.evaluate(combine_generators(test_data), steps=test_steps)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Plot the training results
plt.figure(figsize=(12, 6))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Visualize predictions on test data
test_sample = next(iter(combine_generators(test_data)))  # Take a batch of test data
test_images, test_labels = test_sample

predictions = cnn.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# Plot test sample results
plt.figure(figsize=(12, 12))
for i in range(9):  # Show 9 images
    plt.subplot(3, 3, i+1)
    plt.imshow(test_images[i])
    plt.title(f"True: {categories[true_classes[i]]}\nPred: {categories[predicted_classes[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
