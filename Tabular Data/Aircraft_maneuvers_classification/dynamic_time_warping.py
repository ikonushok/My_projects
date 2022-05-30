import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
from dtw import dtw

from functions import distance_calculation

warnings.filterwarnings("ignore")
pd.pandas.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)
pd.set_option("precision", 2)

# ===================================================================
# Скачаем данные
# ===================================================================
source_root = 'source_root'
source_file = 'IMU_2022_05_14_18_13_42.csv'
df = pd.read_csv(f'{source_root}/{source_file}')
del df['Timestamp[nanosec]']
print(f'Проверим, есть ли NaN в dataframe: {df.isnull().values.any()}')
columns = df.columns.tolist()
labels = pd.unique(df[' label'])
print(f'\n{columns}\n')
# df_grouped = df.groupby(' label')
# print(f"\nСтатистические характеристики классов:\n{df_grouped.agg(['mean', 'std'])}")


# ===================================================================
# Теперь близости различных 2d-матриц данных - https://github.com/pierre-rouanet/dtw
# ===================================================================
df_distances = pd.DataFrame()
for label in labels:
      distances = distance_calculation(df, label)
      df_distances[label] = distances.T
print(df_distances)

plt.figure(figsize=(16, 4))
plt.plot(df_distances)
plt.title(f'Dynamic_Time_Warping of {source_file}')
plt.legend(labels)
plt.show()

plt.figure(figsize=(16, 4))
plt.plot(df_distances.diff())
plt.title(f'Dynamic_Time_Warping of {source_file}')
plt.legend(labels)
plt.show()

sns.heatmap(df_distances.diff().corr())
plt.show()

print(f'\n{"=" * 90}\n'
      f'После расчета методом DTW близостей различных отрезков внутри каждого класса,\n'
      f'мы видим, что маневры легко отличить друг от друга.\n'
      f'Похожи, разве что, маневры 5-7 и 5-8 (что выглядит логичным).'
      f'\n{"=" * 90}\n'
      f'Следовательно, НС должна хорошо обучиться автоматически различать режимы полета в реальном времени.\n'
      f'Для этого можно использовать обычный энкодер-классификатор, или сиамские сети.')
exit()

# =====================================================
# dtw-python
# https://github.com/DynamicTimeWarping/notebooks/blob/master/quickstart/Python.ipynb
# =====================================================
## A noisy sine wave as query
idx = np.linspace(0, 6.28, num=100)
query = np.sin(idx) + np.random.uniform(size=100)/10.0
## A cosine is for template; sin and cos are offset by 25 samples
template = np.cos(idx)
## Find the best match with the canonical recursion formula
alignment = dtw(query, template, keep_internals=True)
## Display the warping curve, i.e. the alignment curve
alignment.plot(type="threeway")

