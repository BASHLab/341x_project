# Lint as: python3
"""Training for the visual wakewords person detection model.

The visual wakewords person detection model is a core model for the TinyMLPerf
benchmark suite. This script provides source for how the reference model was
created and trained, and can be used as a starting point for open submissions
using re-training.

Source: https://github.com/mlcommons/tiny/blob/master/benchmark/training/visual_wake_words/train_vww.py
"""

import os

from absl import app
from vww_model import mobilenet_v1

import tensorflow as tf
assert tf.__version__.startswith('2')

IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 20

BASE_DIR = os.path.join(os.getcwd(), 'vw_coco2014_96')
SPLITS_DIR = os.path.join(os.getcwd(), 'splits')


def load_manifest(manifest_path):
  """Load image paths from manifest file."""
  with open(manifest_path, 'r') as f:
    return [line.strip() for line in f if line.strip()]


def create_generator_from_manifest(manifest_path, augment=False):
  """Create data generator from manifest file."""
  image_paths = load_manifest(manifest_path)
  
  # Create full paths and labels
  filepaths = [os.path.join(BASE_DIR, path) for path in image_paths]
  labels = [0 if path.startswith('non_person/') else 1 for path in image_paths]
  
  if augment:
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=.1,
        horizontal_flip=True,
        rescale=1. / 255)
  else:
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255)
  
  # Create dataframe for flow_from_dataframe
  import pandas as pd
  df = pd.DataFrame({'filename': filepaths, 'class': labels})
  df['class'] = df['class'].astype(str)
  
  generator = datagen.flow_from_dataframe(
      df,
      x_col='filename',
      y_col='class',
      target_size=(IMAGE_SIZE, IMAGE_SIZE),
      batch_size=BATCH_SIZE,
      class_mode='categorical',
      color_mode='rgb',
      shuffle=augment)
  
  return generator


def main(argv):
  if len(argv) >= 2:
    model = tf.keras.models.load_model(argv[1])
  else:
    model = mobilenet_v1()

  model.summary()

  # Load data from manifest files
  train_generator = create_generator_from_manifest(
      os.path.join(SPLITS_DIR, 'train.txt'), augment=True)
  val_generator = create_generator_from_manifest(
      os.path.join(SPLITS_DIR, 'val.txt'), augment=False)
  
  print(train_generator.class_indices)

  model = train_epochs(model, train_generator, val_generator, 20, 0.001)
  model = train_epochs(model, train_generator, val_generator, 10, 0.0005)
  model = train_epochs(model, train_generator, val_generator, 20, 0.00025)

  # Save model HDF5
  if len(argv) >= 3:
    model.save(argv[2])
  else:
    model.save('trained_models/vww_96.h5')


def train_epochs(model, train_generator, val_generator, epoch_count,
                 learning_rate):
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate),
      loss='categorical_crossentropy',
      metrics=['accuracy'])
  history_fine = model.fit(
      train_generator,
      steps_per_epoch=len(train_generator),
      epochs=epoch_count,
      validation_data=val_generator,
      validation_steps=len(val_generator),
      batch_size=BATCH_SIZE)
  return model


if __name__ == '__main__':
  app.run(main)