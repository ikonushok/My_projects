import warnings
warnings.filterwarnings("ignore")

import pandas as pd
pd.pandas.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)
pd.set_option("precision", 2)

from sklearn.model_selection import train_test_split
from functions import TimeSeriesKMeans_cluster_analisys, make_transformations



source_root = 'source_root'
source_file = 'IMU_2022_05_14_18_13_42.csv'
df = pd.read_csv(f'{source_root}/{source_file}')
del df['Timestamp[nanosec]']
columns = df.columns.tolist()
print(f'Проверим, есть ли NaN в dataframe: {df.isnull().values.any()}')
print(f'\n{columns}')

df_grouped = df.groupby(' label')
print(f"\nСтатистические характеристики классов:\n{df_grouped.agg(['mean', 'std'])}")
print('То, что средние и дисперсии классов сильно различаются,\n'
      'позволяет надеяться на то, что можно решить задачу "в лоб" - кластеризацией.')
df_grouped = df_grouped.sum().values
print(df_grouped.shape)

df_x = df[columns[0:-1]]
df_y = df[columns[-1:]]
df_x = make_transformations(df_x)
print(f'\nDataset X:\n{df_x}')
print(f'\nDataset Y:\n{pd.unique(df_y[" label"])}')


# Изучим распределения классов в генеральной совокупности
df_classes_stats = pd.DataFrame()
df_classes_stats['counts'] = df_y[' label'].value_counts(dropna=False).to_frame()
df_classes_stats['share'] = df_y[' label'].value_counts(normalize=True, dropna=False).to_frame()
df_classes_stats.index.rename('classes', inplace=True)
# print(f'\nПостроим распределение классов в генеральной совокупности Y:'
#       f'\n{df_classes_stats}\n')