import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

# ================ DATA PREPARATION STAGE ========================== #

# 1.) Load training and testing images in both coarse and fine labels
(a, coarse_TrLabels), (b, coarse_TsLabels) = keras.datasets.cifar100.load_data(label_mode='coarse')
(fine_train, fine_Trlabels), (fine_test, fine_TsLabels) = keras.datasets.cifar100.load_data(label_mode='fine')

fine_Trlabels = fine_Trlabels.flatten()
fine_TsLabels = fine_TsLabels.flatten()

print('Coarse Classes:', np.unique(coarse_TrLabels))
print('All Fine Classes:', np.unique(fine_Trlabels))

# 2.) Define target custom classes (bicycle, table, telephone, plates, house)
TARGET_FINE_CLASSES = {
    'bicycle':   8,
    'table':     22,
    'telephone': 83,
    'plates':    61,
    'house':     46,
}
target_labels = list(TARGET_FINE_CLASSES.values())
print("\nTarget classes:", TARGET_FINE_CLASSES)

# 3.) Extract training images matching target fine classes
idx_train = np.where(np.isin(fine_Trlabels, target_labels))[0]
train_images = fine_train[idx_train]
train_labels = fine_Trlabels[idx_train].copy()
print(f'\nTotal images from TRAINING DATASET: {len(idx_train)}')
print('Shape of training dataset:', train_images.shape)
print('Unique fine classes found:', np.unique(train_labels))

# 4.) Extract testing images matching target fine classes
idx_test = np.where(np.isin(fine_TsLabels, target_labels))[0]
test_images = fine_test[idx_test]
test_labels = fine_TsLabels[idx_test].copy()
print(f'\nTotal images from TESTING DATASET: {len(idx_test)}')
print('Shape of testing dataset:', test_images.shape)
print('Unique fine classes found:', np.unique(test_labels))

# 5.) Relabel training and testing dataset to start from zero (0)
sorted_labels = sorted(target_labels)  # [8, 22, 46, 61, 83]
label_map = {old: new for new, old in enumerate(sorted_labels)}

for old, new in label_map.items():
    train_labels[train_labels == old] = new
    test_labels[test_labels == old] = new

print("\nLabel remapping:")
for name, old_idx in TARGET_FINE_CLASSES.items():
    print(f"  {name:12s} (original={old_idx}) → new label {label_map[old_idx]}")

# 6.) Plot few samples from the TESTING DATASET
class_names = ['bicycle', 'table', 'telephone', 'plates', 'house']  # sorted by index

plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[test_labels[i]])
plt.show()
