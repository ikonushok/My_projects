import os
import random

import numpy as np
from collections import Counter
from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization, Flatten,
                                     Conv1D, MaxPooling1D, RepeatVector)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tqdm import trange

from Aircraft_maneuvers_classification.constants import source_root, path_outputs
from Aircraft_maneuvers_classification.utilits.create_dataset_functions import read_data


# сделаем так, чтобы tf не резервировал под себя сразу всю память
# https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    # фиксируем сиды
    # https://coderoad.ru/51249811/Воспроизводимые-результаты-в-Tensorflow-с-tf-set_random_seed
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
    tf.random.set_seed(seed)



# parameters
patch = 30
batch_size = 256

# скачаем сбалансированный датасет
arr_x, arr_y = read_data(source_root, patch=patch)
arr_y = arr_y - 1  # тк у нас нумерация классов начинается с 0
num_classes = len(np.unique(arr_y))
print(arr_x.shape, arr_y.shape, np.unique(arr_y), num_classes)

# Scaling the data
print(f'\nData before scaling:\n{arr_x[0][0]}')
scaler = StandardScaler()
scaled_x = scaler.fit_transform(arr_x.reshape(arr_x.shape[0], arr_x.shape[1] * arr_x.shape[2]))
x_train = scaled_x.reshape(arr_x.shape[0], arr_x.shape[1], arr_x.shape[2])
print(f'Data after scaling:\n{x_train[0][0]}\n')

# превожу в to_categorical
from tensorflow.keras.utils import to_categorical
categorical_labels = to_categorical(arr_y, num_classes=num_classes)
print(f'Образец категориального Y:\t{categorical_labels[0]}\n')

# Разделим на тренировочную и обучающую выборки
x_, x_val, y_, y_val = train_test_split(arr_x, categorical_labels, test_size=0.2, shuffle=True)
x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=0.1, shuffle=True)
print(f'Train:\tX {x_train.shape},\tY {y_train.shape}\n'
      f'Val:\tX {x_val.shape},\tY {y_val.shape}\n'
      f'Test:\tX {x_test.shape},\tY {y_test.shape}')


""" Мodelling """
# x_train.shape[1] - число шагов назад для обучения
# x_train.shape[2] - число столбцов в обучающей выборке
drop = 0.4
input = Input(shape=(x_train.shape[1], x_train.shape[2]))
x = Flatten()(input)
x = Dense(patch*20, activation='relu')(x)
x = Dropout(drop)(x)
x = Dense(num_classes, activation='softmax')(x)
model = Model(input, x)

# print(model.summary()) #Вывод структуры модели
# plot_model(model, dpi=50, show_shapes=True, show_layer_names=True)

cyclical_learning_rate = tfa.optimizers.CyclicalLearningRate(
                initial_learning_rate=1e-5,  # [5e-5]
                maximal_learning_rate=1e-3,  # [1e-3]
                scale_fn=lambda x: 1 / (2. ** (x - 1)),
                scale_mode='cycle',  # 'cycle', 'iteration'
                step_size=5  # от 2 до 10 [2]
)
optimizer = tf.keras.optimizers.Adam(cyclical_learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['AUC', 'categorical_accuracy'])
print('Model compiled...')

# коллбэки
early_stopping = EarlyStopping(monitor='val_accuracy', patience=4, verbose=1, restore_best_weights=True, mode='max')
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=4, min_lr=1e-07, verbose=1)
checkpoint = ModelCheckpoint(f'{path_outputs}/simple_nn.h5',
                             monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test),
                    verbose=1,
                    # callbacks=[early_stopping, reduce_lr, checkpoint]
                    )


plt.figure(figsize=(6, 4))
plt.plot(history.history['val_categorical_accuracy'], label='val_categorical_accuracy')
plt.plot(history.history['val_auc'], label='val_auc')
# plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label='val_loss')
plt.title(f'Параметры обучения модели для:\npatch = {patch},  batch_size = { batch_size}\ncyclical_learning_rate')
plt.legend()
plt.grid()
plt.savefig(f'{path_outputs}/history_bs_{batch_size}_patch_{patch}_clr.png')
plt.show()

print('\nValidation AUC achieved:', (history.history['val_auc'])[-1])
print('Validation accuracy achieved:', (history.history['val_categorical_accuracy'])[-1])


# Распознавания всех тестовых вариантов и вывода класса
arr_test = []
true = 0
for i in trange(x_test.shape[0], desc='Проверяем модель для тестовой выборки...'):
    x = x_test[i]
    x = np.expand_dims(x, axis=0)
    prediction = model.predict(x)  # Распознаём наш пример
    prediction = np.argmax(prediction)  # Получаем индекс самого большого элемента (это итоговая цифра)

    if prediction == np.argmax(y_test[i]):
        arr_test.append('True')
        true += 1
    else:
        arr_test.append('False')

print(f'{Counter(arr_test)}\t|\taccuracy = {true/x_test.shape[0]}')