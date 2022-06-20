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

# fig = plt.figure(figsize=(8,5))
# ax = sns.barplot(x=df.Sentiment.unique(),y=df.Sentiment.value_counts());
# ax.set(xlabel='Labels')
# plt.show()

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
print(f'80 Tokenized words:\t{reviews_ints[0][:20]}\t{len(reviews_ints[0])}')

# MAKE ALL REVIEWS SAME LENGTH
## outlier review stats
rewiew_lens = Counter([len(x) for x in reviews])
print(f'{rewiew_lens[0]} - {max(rewiew_lens)}')
plt.hist(rewiew_lens)
plt.title('Распределение топиков по длине (число слов)')
plt.show()


def pad_text(encoded_reviews, seq_length):
    reviews = []
    for review in encoded_reviews:
        if len(review) >= seq_length:
            reviews.append(review[:seq_length])
        else:
            reviews.append([0] * (seq_length - len(review)) + review)
    return np.array(reviews)

encoded_reviews = [[vocab_to_int.get(word) for word in review.split()] for review in reviews]
encoded_labels = np.array( [label for idx, label in enumerate(labels) if len(encoded_reviews[idx]) > 0] )
encoded_reviews = [review for review in encoded_reviews if len(review) > 0]
padded_reviews = pad_text(encoded_reviews, seq_length=200)
print('\nПосмотрим на образцы закодированных топиков:')
print(padded_reviews[0], reviews[0])
print(padded_reviews[1], reviews[1])


# SPLIT DATA & GET (REVIEW, LABEL) DATALOADER
train_ratio = 0.8
valid_ratio = (1 - train_ratio)/2
total = padded_reviews.shape[0]
train_cutoff = int(total * train_ratio)
valid_cutoff = int(total * (1 - valid_ratio))

train_x, train_y = padded_reviews[:train_cutoff], encoded_labels[:train_cutoff]
valid_x, valid_y = padded_reviews[:train_cutoff : valid_cutoff], encoded_labels[train_cutoff : valid_cutoff]
test_x, test_y = padded_reviews[valid_cutoff:], encoded_labels[valid_cutoff:]

from torch.utils.data import TensorDataset, DataLoader

train_data = TensorDataset(train_x, train_y)
valid_data = TensorDataset(valid_x, valid_y)
test_data = TensorDataset(test_x, test_y)

batch_size = 50
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)