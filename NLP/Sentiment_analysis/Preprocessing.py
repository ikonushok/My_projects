# https://colab.research.google.com/drive/1xqkvuNDg0Opk-aZpicTVPNY4wUdC7Y7v?usp=sharing#scrollTo=Tl4_dITCUk83
# https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b
import re
import warnings

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
pd.pandas.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)
pd.set_option("precision", 2)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device = ', device)


# Parameters
source_root = 'source_root'
filename = 'dataset.csv'
destination_folder = 'outputs'
train_test_ratio = 0.10
train_valid_ratio = 0.80
first_n_words = 200


# Preprocessing
def trim_string(x):

    x = x.split(maxsplit=first_n_words)
    x = ' '.join(x[:first_n_words])

    return x


# Read raw data
df = pd.read_csv(f'{source_root}/{filename}', delimiter=';')
df = df[['Sentiment', 'SentimentText']]
df.dropna(inplace=True)
# Split according to label
df1 = df[df['Sentiment'] == 1]
df2 = df[df['Sentiment'] == 2]
df2['Sentiment'] = 0
df = pd.concat([df1, df2], ignore_index=True, sort=False)
print(df)

# Prepare columns
# df_raw['Sentiment'] = (df_raw['Sentiment'] == 'FAKE').astype('int')
# df_raw['titletext'] = df_raw['title'] + ". " + df_raw['text']
# df_raw = df_raw.reindex(columns=['label', 'title', 'text', 'titletext'])

# Drop rows with empty text
df.drop(df[df.SentimentText.str.len() < 5].index, inplace=True)
print(df.shape)

fig = plt.figure(figsize=(8, 5))
ax = sns.barplot(x=df.Sentiment.unique(), y=df.Sentiment.value_counts())
ax.set(xlabel='Labels')
plt.title('Распределение классов')
plt.show()


# Drop rows with empty text
df.drop(df[df.SentimentText.str.len() < 5].index, inplace=True)
print(df.shape)
print(df)
print(type(df.SentimentText))
print(f'\nУникальных значений:\n{df.nunique()}')

fig = plt.figure(figsize=(8,5))
ax = sns.barplot(x=df.Sentiment.unique(), y=df.Sentiment.value_counts())
ax.set(xlabel='Labels')
plt.show()

# Сохраним метки в виде списков
reviews = df.SentimentText.values.tolist()
labels = df.Sentiment.values.tolist()

# Удалим заглавные и знаки препинания и составим словарь
for i, review in enumerate(reviews):
    review = str(review).lower()
    review = re.sub(r'[^\w\s]', '', review)
    reviews[i] = review
print(f'60 reviews:\t{reviews[0][:70]}, {type(reviews)}, {len(reviews)}')
print(f'61 labels:\t{labels[0]}, {type(labels)}, {len(labels)}')

# Создадим очищенный от препинаний и тд датасет
df_new = pd.DataFrame(list(zip(reviews, labels)), columns=['text', 'label'])
print(df_new)


# Trim text and titletext to first_n_words
# df_raw['SentimentText'] = df_raw['SentimentText'].apply(trim_string)
# df_raw['titletext'] = df_raw['titletext'].apply(trim_string)

# Split according to label
df_pos = df_new[df_new['label'] == 1]
df_neg = df_new[df_new['label'] == 0]

# Train-test split
df_pos_full_train, df_pos_test = train_test_split(df_pos, train_size=train_test_ratio, random_state=1)
df_neg_full_train, df_neg_test = train_test_split(df_neg, train_size=train_test_ratio, random_state=1)

# Train-valid split
df_pos_train, df_pos_valid = train_test_split(df_pos_full_train, train_size=train_valid_ratio, random_state=1)
df_neg_train, df_neg_valid = train_test_split(df_neg_full_train, train_size=train_valid_ratio, random_state=1)

# Concatenate splits of different labels
df_train = pd.concat([df_pos_train, df_neg_train], ignore_index=True, sort=False)
df_valid = pd.concat([df_pos_valid, df_neg_valid], ignore_index=True, sort=False)
df_test = pd.concat([df_pos_test, df_neg_test], ignore_index=True, sort=False)

# Write preprocessed data
df_train.to_csv(destination_folder + '/train.csv', index=False)
df_valid.to_csv(destination_folder + '/valid.csv', index=False)
df_test.to_csv(destination_folder + '/test.csv', index=False)