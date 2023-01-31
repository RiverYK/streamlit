
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
import tensorflow as tf
import pathlib
import glob

from PIL import ImageShow
from tensorflow import keras
from keras import layers
from keras.models import Sequential

from keras.utils import load_img, img_to_array
from keras.layers.preprocessing.image_preprocessing import Rescaling

my_data = pd.read_csv('./bee_classification/bee_data.csv')
dataset_path = './bee_classification/bee_imgs_classified/bee_imgs'
data_dir = pathlib.Path(dataset_path)
# print(data_dir)
# print(type(data_dir))

# /* : 폴더 내 모든 파일
healthy = list(data_dir.glob('healthy/*'))
# print(healthy)
PIL.Image.open(str(healthy[0]))

# bee_images 폴더에 사진이 몇개 있는지 확인
# image_count = len(list(data_dir.glob('*.png')))
# print(image_count)

# CSV의 카테고리당 사진 개수
# print(my_data['health'].value_counts())

# Dataset 생성

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names

# print(class_names)

# Dataset 시각화

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
#         plt.show()

# configure dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# standardize dataset

normalization_layer = Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

# model 생성

num_classes = 6

model = Sequential([
  Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# compile data

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# training

# epochs=10
epochs = 1
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# <<visualizing>>


# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()


# 샘플 사진 입력
# bee_img = './bee_classification/bee_imgs_classified\Predictions/bee_drawing.png'

# 꿀벌 상태
def bee_classify(bee_img):
    img = load_img(
        bee_img, target_size=(img_height, img_width)
    )

    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # print(
    #     "사진의 벌은 {:.2f}의 확률로 {}인 상태입니다."
    #     .format(100 * np.max(score), class_names[np.argmax(score)])
    # )

    percentage = round(100 * np.max(score), 2)
    bee_state = class_names[np.argmax(score)]
    # print (f'사진의 벌은 {percentage}의 확률로 {bee_state}인 상태입니다.')
    
    return percentage, bee_state