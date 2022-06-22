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
from torch.utils.data import TensorDataset, DataLoader

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
# Split according to label
df1 = df[df['Sentiment'] == 1]
df2 = df[df['Sentiment'] == 2]
df = pd.concat([df1, df2], ignore_index=True, sort=False)
# Drop rows with empty text
df.drop(df[df.SentimentText.str.len() < 5].index, inplace=True)
print(df.shape)
print(df)
print(type(df.SentimentText))
print(f'\nУникальных значений:\n{df.nunique()}')

fig = plt.figure(figsize=(8,5))
ax = sns.barplot(x=df.Sentiment.unique(),y=df.Sentiment.value_counts());
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
print(f'61 reviews:\t{reviews[0][:60]}, {type(reviews)}, {len(reviews)}')
print(f'62 labels:\t{labels[0]}, {type(labels)}, {len(labels)}')
all_text = ''.join(c for c in reviews if c not in punctuation)
print(f'64 all_text:\t{all_text[:60]}, {type(all_text)}, {len(all_text)}')
review_split = all_text.split('\n')
print(f'66 review_split:\t{review_split[0][:60]}, {type(review_split)}, {len(review_split[0])}')
words = all_text.split()
print(f'68 words:\t{words[:10]}, {type(words)}, {len(words)}')

# Encoding the words
## Build a dictionary that map words to integers
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: w for w, word in enumerate(vocab, 1)}
print(f'74 Unique words:\t{vocab[:20]}\t{len(vocab_to_int)}')

## use the dict to tokenize each review in review_split
reviews_ints = []
for review in review_split:
    reviews_ints.append([vocab_to_int[word] for word in review.split()])
print(f'80 Tokenized words (reviews_ints):\t{reviews_ints[0][:20]}\t{len(reviews_ints[0])}')

# MAKE ALL REVIEWS SAME LENGTH
## outlier review stats
rewiew_lens = Counter([len(x) for x in reviews])
print(f'{rewiew_lens[0]} - {max(rewiew_lens)}')
plt.hist(rewiew_lens)
plt.title('Распределение топиков по длине (число слов)')
plt.show()



def pad_features(reviews_ints, seq_length):
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    return features

seq_lenght = 200
# encoded_reviews = [[vocab_to_int.get(word) for word in review.split()] for review in reviews]
# encoded_labels = np.array( [label for idx, label in enumerate(labels) if len(encoded_reviews[idx]) > 0] )
# encoded_reviews = [review for review in encoded_reviews if len(review) > 0]
features = pad_features(reviews_ints, seq_length=seq_lenght)
assert len(features) == len(reviews_ints), 'Your features should have as many rows as reviews'
assert len(features[0] == seq_lenght), 'Each feature row should contain seq_lenght values'
print('\nПосмотрим на образцы закодированных топиков:')
print(features, features.shape, len(features), reviews[0])


# SPLIT DATA & GET (REVIEW, LABEL) DATALOADER
split_frac = 0.8

## split data into train, val, test
split_idx = int(len(features)*split_frac)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = labels[:split_idx], labels[split_idx:]

test_idx = int(len(remaining_x)*0.5)
val_x, test_x = features[:test_idx], features[test_idx:]
val_y, test_y = labels[:test_idx], labels[test_idx:]


print(f'\n'
      f'train_x: {train_x.shape}\ttrain_y: {len(train_y)}\t{type(train_x)}, {type(train_y)}\n'
      f'val_x: {val_x.shape}\ttrain_y: {len(val_y)}\n'
      f'test_x: {test_x.shape}\ttest_y: {len(test_y)}\n')

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(val_x, val_y)
test_data = TensorDataset(test_x, test_y)

batch_size = 50
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
print(f'\n'
      f'train_x:\t{train_x.shape}\ttrain_y:\t{train_y.shape}\n'
      f'valid_x:\t{val_x.shape}\ttrain_y:\t{val_y.shape}\n'
      f'test_x:\t{test_x.shape}\ttest_y:\t{test_y.shape}\n')
