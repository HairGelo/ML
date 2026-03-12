import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

# ================ DATA PREPARATION STAGE ========================== #
# 1.) Load data
(a, coarse_TrLabels), (b, coarse_TsLabels) = keras.datasets.cifar100.load_data(label_mode='coarse')
(fine_train, fine_Trlabels), (fine_test, fine_TsLabels) = keras.datasets.cifar100.load_data(label_mode='fine')

coarse_TrLabels = coarse_TrLabels.flatten()
coarse_TsLabels = coarse_TsLabels.flatten()
fine_Trlabels   = fine_Trlabels.flatten()
fine_TsLabels   = fine_TsLabels.flatten()

# 2.) Define superclass → specific fine class name to extract
# Each superclass has 5 fine classes; we only want 1 from each
SUPERCLASS_MAP = {
    'bicycle':   19,   # vehicles 1
    'table':      8,   # household furniture
    'telephone':  5,   # household electrical devices
    'plates':     3,   # food containers
    'house':     10,   # large man-made outdoor things
}

# 3.) For each superclass, extract ONLY the one fine class we want
def extract_one_class(class_name, coarse_idx, coarse_labels, fine_images, fine_labels):
    # Get all images belonging to this superclass
    superclass_idx = np.where(coarse_labels == coarse_idx)[0]
    superclass_images = fine_images[superclass_idx]
    superclass_labels = fine_labels[superclass_idx]

    # Find the unique fine labels in this superclass
    unique_fine = np.unique(superclass_labels)
    print(f"\n[{class_name}] Superclass {coarse_idx} fine labels: {unique_fine}")

    # Pick the fine label that corresponds to our target class
    # Since we can now SEE the fine labels, pick the correct one
    target_fine = unique_fine[CLASS_FINE_PICK[class_name]]
    print(f"  → Picking fine label index [{CLASS_FINE_PICK[class_name]}] = {target_fine}")

    idx = np.where(superclass_labels == target_fine)[0]
    return superclass_images[idx], superclass_labels[idx], target_fine

# 4.) Fine label position within each superclass (0-4, alphabetical order)
# vehicles 1:               bicycle(0), bus(1), motorcycle(2), pickup_truck(3), train(4)
# household furniture:      bed(0), chair(1), couch(2), table(3), wardrobe(4)
# household electrical:     clock(0), keyboard(1), lamp(2), telephone(3), television(4)
# food containers:          bottles(0), bowls(1), cans(2), cups(3), plates(4)
# large man-made outdoor:   bridge(0), castle(1), house(2), road(3), skyscraper(4)
CLASS_FINE_PICK = {
    'bicycle':   0,   # first in vehicles 1
    'table':     3,   # fourth in household furniture
    'telephone': 3,   # fourth in household electrical devices
    'plates':    4,   # fifth in food containers
    'house':     2,   # third in large man-made outdoor things
}

# 5.) Extract and concatenate all classes for TRAINING
all_train_images, all_train_labels = [], []
for new_label, (class_name, coarse_idx) in enumerate(SUPERCLASS_MAP.items()):
    imgs, _, _ = extract_one_class(class_name, coarse_idx, coarse_TrLabels, fine_train, fine_Trlabels)
    all_train_images.append(imgs)
    all_train_labels.append(np.full(len(imgs), new_label))  # relabel 0-4

train_images = np.concatenate(all_train_images, axis=0)
train_labels = np.concatenate(all_train_labels, axis=0)
print(f'\nTraining dataset shape: {train_images.shape}')
print(f'Training labels: {np.unique(train_labels)}')

# 6.) Extract and concatenate all classes for TESTING
all_test_images, all_test_labels = [], []
for new_label, (class_name, coarse_idx) in enumerate(SUPERCLASS_MAP.items()):
    imgs, _, _ = extract_one_class(class_name, coarse_idx, coarse_TsLabels, fine_test, fine_TsLabels)
    all_test_images.append(imgs)
    all_test_labels.append(np.full(len(imgs), new_label))

test_images = np.concatenate(all_test_images, axis=0)
test_labels = np.concatenate(all_test_labels, axis=0)
print(f'\nTesting dataset shape: {test_images.shape}')
print(f'Testing labels: {np.unique(test_labels)}')

# 7.) Plot samples from testing dataset (one per class)
class_names = list(SUPERCLASS_MAP.keys())  # ['bicycle', 'table', 'telephone', 'plates', 'house']
plt.figure(figsize=(10, 2))
for i in range(5):
    idx = np.where(test_labels == i)[0][0]  # grab first image of each class
    plt.subplot(1, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[idx])
    plt.xlabel(class_names[i])
plt.suptitle('One sample per class')
plt.show()

# ================ MODEL BUILDING STAGE ========================== #
# 7.) Normalize images to [0, 1] range
train_images = train_images / 255.0
test_images  = test_images  / 255.0

# 8.) Build the model
model = tf.keras.Sequential()
# 32 convolution filters each of size 3x3
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
# Choose best features via pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# Flatten for classification output
model.add(Flatten())
# Fully connected layer
model.add(Dense(128, activation='relu'))
# Output layer — 5 classes (bicycle, table, telephone, plates, house)
model.add(Dense(len(SUPERCLASS_MAP, activation='softmax')))
model.summary()

# 9.) Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# 10.) Train the model
metricInfo = model.fit(train_images, train_labels, epochs=10, validation_split=0.1)

# 11.) Plot Training vs Validation Loss
loss     = metricInfo.history['loss']
val_loss = metricInfo.history['val_loss']
epochs   = range(1, len(loss) + 1)

plt.clf()
plt.plot(epochs, loss, 'g-', label="Training loss")
plt.plot(epochs, val_loss, 'b-', label='Validation loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 12.) Evaluate on test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest Accuracy: {test_acc:.4f}')
print(f'Test Loss:     {test_loss:.4f}')

input("Press Enter to exit...")