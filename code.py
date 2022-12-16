import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from sklearn.manifold import SpectralEmbedding

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

plt.figure()
plt.suptitle("MNIST Dataset Samples", fontsize = 'x-large')
label_indexes = { training_labels[i]: i for i in range(len(training_labels)) }

for i in range(9):
    index = label_indexes[i]
    plt.subplot(3, 3, i + 1)
    plt.title(training_labels[index])
    plt.imshow(training_images[index], cmap = 'Greys')
    
plt.tight_layout()

processed_training_images = training_images / 255.0
processed_test_images = test_images / 255.0

label_set = np.sort(np.unique(training_labels))
training_one_hots = keras.utils.to_categorical(training_labels, len(label_set))
test_one_hots = keras.utils.to_categorical(test_labels, len(label_set))
print("Sample One-Hots:")

for i in range(9):
    print(f"{training_labels[i]}: {training_one_hots[i]}")
    
model1 = Sequential()
model1.add(Dense(512, input_shape = (28 * 28,), activation = "relu"))           
model1.add(Dropout(0.15))
model1.add(Dense(512, activation = "relu"))
model1.add(Dropout(0.15))
model1.add(Dense(10, activation = "softmax"))
model1.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model1.summary()

# Reshape data for model
training_vectors1 = processed_training_images.reshape(len(processed_training_images), 28 * 28)
test_vectors1 = processed_test_images.reshape(len(processed_test_images), 28 * 28)

model1_path = "mnist_model1.saved_model"
model1_history_path = "mnist_model1.saved_model_history"

if os.path.exists(model1_path) and os.path.exists(model1_history_path):
    # Load trained model
    model1 = keras.models.load_model(model1_path)
    history1 = pickle.load(open(model1_history_path, "rb"))
else:
    # Train new model
    tensorflow.random.set_seed(12345)
    model1.fit(training_vectors1, training_one_hots,
               batch_size = 64,
               epochs = 5,
               verbose = 1,
               validation_data = (test_vectors1, test_one_hots))
    history1 = model1.history.history
    model1.save(model1_path)
    pickle.dump(history1, open(model1_history_path, "wb"))
    
print(f"Training Accuracy: {history1['accuracy'][-1]:.4}")
print(f"Validation Accuracy: {history1['val_accuracy'][-1]:.4}")

plt.figure()
plt.title("Model 1 Accuracies")
plt.plot(history1["accuracy"], marker = "o", label = "Training Accuracy")
plt.plot(history1["val_accuracy"], marker = "o", label = "Validation Accuracy")
plt.legend()
plt.grid()

model2 = Sequential()
model2.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1)))
model2.add(Conv2D(32, (3, 3), activation = "relu"))
model2.add(MaxPooling2D(pool_size = (2, 2)))
model2.add(Dropout(0.2))
model2.add(Flatten())
model2.add(Dense(128, activation = "relu"))
model2.add(Dropout(0.2))
model2.add(Dense(10, activation = "softmax"))
model2.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model2.summary()

# Reshape data for model
training_vectors2 = training_images.reshape(len(training_images), 28, 28, 1)
test_vectors2 = test_images.reshape(len(test_images), 28, 28, 1)

model2_path = "mnist_model2.saved_model"
model2_history_path = "mnist_model2.saved_model_history"

if os.path.exists(model2_path) and os.path.exists(model2_history_path):
    # Load trained model
    model2 = keras.models.load_model(model2_path)
    history2 = pickle.load(open(model2_history_path, "rb"))
else:
    # Train new model
    tensorflow.random.set_seed(12345)
    model2.fit(training_vectors2, training_one_hots,
               batch_size = 64,
               epochs = 5,
               verbose = 1,
               validation_data = (test_vectors2, test_one_hots))
    history2 = model2.history.history
    model2.save(model2_path)
    pickle.dump(history2, open(model2_history_path, "wb"))

print(f"Training Accuracy: {history2['accuracy'][-1]:.4}")
print(f"Validation Accuracy: {history2['val_accuracy'][-1]:.4}")

plt.figure()
plt.title("Model 2 Accuracies")
plt.plot(history2["accuracy"], marker = "o", label = "Training Accuracy")
plt.plot(history2["val_accuracy"], marker = "o", label = "Validation Accuracy")
plt.legend()
plt.grid()

best_model_index = np.argmax([x["val_accuracy"][-1] + x["accuracy"][-1] for x in (history1, history2)])
best_model = (model1, model2)[best_model_index]
history = (history1, history2)[best_model_index]
test_vectors = (test_vectors1, test_vectors2)[best_model_index]

print(f"Best Model: {best_model_index + 1}")
print(f"Training Accuracy: {history['accuracy'][-1]:.4}")
print(f"Validation Accuracy: {history['val_accuracy'][-1]:.4}")

# Predict the class labels
predictions = best_model.predict(test_vectors)
predicted_labels = predictions.argmax(axis = -1)
correct_filter = predicted_labels == test_labels
correct_predictions = np.flatnonzero(correct_filter)
incorrect_predictions = np.flatnonzero(~correct_filter)

# Plot sample of correct predictions
plt.figure()
plt.suptitle("Correct Predictions", fontsize = 'x-large')

for i in range(9):
    index = correct_predictions[i]
    plt.subplot(3, 3, i + 1)
    plt.title(f"Class {test_labels[index]}\nPredicted {predicted_labels[index]}")
    plt.imshow(test_images[index], cmap = 'Greys')
    
plt.tight_layout()

# Plot sample of incorrect predictions
plt.figure()
plt.suptitle("Incorrect Predictions", fontsize = 'x-large')

for i in range(9):
    index = incorrect_predictions[i]
    plt.subplot(3, 3, i + 1)
    plt.title(f"Class {test_labels[index]}\nPredicted {predicted_labels[index]}")
    plt.imshow(test_images[index], cmap = 'Greys')

plt.tight_layout()
