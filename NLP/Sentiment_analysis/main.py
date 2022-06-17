import pathlib
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from string import punctuation

import torch
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm, trange

tqdm.pandas(desc='Progress')
warnings.filterwarnings('ignore')
pd.pandas.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)
pd.set_option("precision", 2)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device = ', device)


# Process Data through pytorch Dataset
source_root = 'source_root'
filename = 'dataset.csv'
# load csv in pandas dataframe
# df = pd.read_csv(data_root / 'Sentiment Analysis Dataset.csv', error_bad_lines=False)
df = pd.read_csv(f'{source_root}/{filename}', delimiter=';')
df = df[['Sentiment', 'SentimentText']]
# df = df.query("Sentiment != 1")
print(df)
print(type(df.SentimentText))
print(f'\nУникальных значений:\n{df.nunique()}')

# split the data into train and validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[['Sentiment']])
train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

print(f'train_df: {train_df.shape},\tval_df: {val_df.shape}')

fig = plt.figure(figsize=(8,5))
ax = sns.barplot(x=df.Sentiment.unique(),y=df.Sentiment.value_counts());
ax.set(xlabel='Labels')
plt.show()


# Сохраним метки в виде списков
values = df.Sentiment.values.tolist()
print('value[0] =', values[0])
reviews = df.SentimentText.values.tolist()
reviews = ''.join(str(reviews))
reviews = reviews.lower()
print(reviews[:60], type(reviews), len(reviews))

all_text = re.sub(r'[^\w\s]','', reviews)
print(all_text[:60], type(all_text), len(all_text))
words = all_text.split()
print(words[:10])


