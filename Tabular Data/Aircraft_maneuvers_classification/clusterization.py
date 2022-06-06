import warnings

from constants import source_root

warnings.filterwarnings("ignore")

import pandas as pd
pd.pandas.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)
pd.set_option("precision", 2)

from sklearn.model_selection import train_test_split
from utilits.create_dataset_functions import TimeSeriesKMeans_cluster_analisys, make_transformations


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

# Для ускорения перемешаем и кластеризуем 10% при условии соблюдения пропорций классов
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.02,
                                                    shuffle=True, random_state=42)
# Изучим распределения классов в отобранной совокупности
df_classes_stats['counts~'] = y_test[' label'].value_counts(dropna=False).to_frame()
df_classes_stats['share~'] = y_test[' label'].value_counts(normalize=True, dropna=False).to_frame()
df_classes_stats.index.rename('classes', inplace=True)
print(f'\nПостроим распределение классов в отобранной совокупности Y:'
      f'\n{df_classes_stats}')


#https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html
TimeSeriesKMeans_params = {'n_init': 1, 'max_iter_barycenter': 15, 'n_jobs': 25, 'metric': 'softdtw'}
range_n_clusters = [i for i in range(2, df_classes_stats.shape[0]+1)]
best_cluster_labels = TimeSeriesKMeans_cluster_analisys(X_test,
                                                        range_n_clusters,
                                                        TimeSeriesKMeans_params)
y_test['new classes'] = best_cluster_labels
print(f'\nНовая разметка Y:\n{y_test}')
print(f'\n{"=" * 90}\nВывод:\nРешить задачу кластеризацией не получилось!\n{"=" * 90}')