import os
import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from constants import path_outputs, source_root
from utilits.create_dataset_functions import make_transformations, split_sequence, read_data

warnings.filterwarnings("ignore")
pd.pandas.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)
# pd.set_option("precision", 2)

source_file = 'IMU_2022_05_14_18_13_42.csv'
df = pd.read_csv(f'{source_root}/{source_file}')
del df['Timestamp[nanosec]']
columns = df.columns.tolist()
columns = columns[-1:] + columns[:-1]
df = df[columns]
df = make_transformations(df)
columns = df.columns.tolist()
print(f'Проверим, есть ли NaN в dataframe: {df.isnull().values.any()}')

df_x = df[columns[1:]]
df_y = df[columns[:1]]
print(f'\nDataset X:\n{df_x}')
print(f'\nDataset Y:\n{pd.unique(df_y[" label"])}')
# # Изучим распределения классов в генеральной совокупности
df_classes_stats = pd.DataFrame()
df_classes_stats['counts'] = df_y[' label'].value_counts(dropna=False).to_frame()
df_classes_stats['share'] = df_y[' label'].value_counts(normalize=True, dropna=False).to_frame()
df_classes_stats.index.rename('classes', inplace=True)
print(f'\nПостроим распределение классов в отобранной совокупности Y:\n{df_classes_stats}\n')
plt.hist(df_y)
plt.title('Распределение классов в первоначальных данных')
plt.savefig(f'{path_outputs}/unbalanced_dataset.png')
plt.show()


# Выберем в середине каждого класса 10000 строк (чтобы устранить дисбаланс классов)
maneuver1 = df.loc[df[' label'] == 1]
maneuver1 = maneuver1[maneuver1.shape[0]//2 - 5000:maneuver1.shape[0]//2 + 5000]
maneuver1.to_csv(f'{source_root}/maneuver1.csv', index=False)
maneuver2 = df.loc[df[' label'] == 2]
maneuver2 = maneuver2[maneuver2.shape[0]//2 - 5000:maneuver2.shape[0]//2 + 5000]
maneuver2.to_csv(f'{source_root}/maneuver2.csv', index=False)
maneuver3 = df.loc[df[' label'] == 3]
maneuver3 = maneuver3[maneuver3.shape[0]//2 - 5000:maneuver3.shape[0]//2 + 5000]
maneuver3.to_csv(f'{source_root}/maneuver3.csv', index=False)
maneuver4 = df.loc[df[' label'] == 4]
maneuver4 = maneuver4[maneuver4.shape[0]//2 - 5000:maneuver4.shape[0]//2 + 5000]
maneuver4.to_csv(f'{source_root}/maneuver4.csv', index=False)
maneuver5 = df.loc[df[' label'] == 5]
maneuver5 = maneuver5[maneuver5.shape[0]//2 - 5000:maneuver5.shape[0]//2 + 5000]
maneuver5.to_csv(f'{source_root}/maneuver5.csv', index=False)
maneuver6 = df.loc[df[' label'] == 6]
maneuver6 = maneuver6[maneuver6.shape[0]//2 - 5000:maneuver6.shape[0]//2 + 5000]
maneuver6.to_csv(f'{source_root}/maneuver6.csv', index=False)
maneuver7 = df.loc[df[' label'] == 7]
maneuver7 = maneuver7[maneuver7.shape[0]//2 - 5000:maneuver7.shape[0]//2 + 5000]
maneuver7.to_csv(f'{source_root}/maneuver7.csv', index=False)
maneuver8 = df.loc[df[' label'] == 8]
maneuver8 = maneuver8[maneuver8.shape[0]//2 - 5000:maneuver8.shape[0]//2 + 5000]
maneuver8.to_csv(f'{source_root}/maneuver8.csv', index=False)
maneuver9 = df.loc[df[' label'] == 9]
maneuver9 = maneuver9[maneuver9.shape[0]//2 - 5000:maneuver9.shape[0]//2 + 5000]
maneuver9.to_csv(f'{source_root}/maneuver9.csv', index=False)
maneuver10 = df.loc[df[' label'] == 10]
maneuver10 = maneuver10[maneuver10.shape[0]//2 - 5000:maneuver10.shape[0]//2 + 5000]
maneuver10.to_csv(f'{source_root}/maneuver10.csv', index=False)

file_list = os.listdir(source_root)
file_list.sort()
if '.DS_Store' in file_list:
    file_list.remove('.DS_Store')
if 'IMU_2022_05_14_18_13_42.csv' in file_list:
    file_list.remove('IMU_2022_05_14_18_13_42.csv')
print(file_list)

# Разделение массива на выборки для обучения нейросети
# sequence = строка массива df
# patch = число баров в прошлом для анализа - необходимо найти оптимальный!!
# forvard_lag = предсказание какого бара в будущем делаем


arr_x, arr_y = read_data(source_root, patch=10)
print(arr_x.shape, arr_y.shape)

#Train & test data split
x_train, x_test, y_train, y_test = train_test_split(arr_x, arr_y, test_size=0.2, shuffle=True)
print(f'\nСбалансированный датасет:\n'
      f'Train:\tX shape {x_train.shape},\tY shape {y_train.shape}\n'
      f'Test:\tX shape {x_test.shape},\tY shape {y_test.shape}')
plt.hist(y_train)
plt.hist(y_test)
plt.title('Распределение классов в сбалансированном датасете')
plt.savefig(f'{path_outputs}/balanced_dataset.png')
plt.show()
