# https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130


import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pathlib
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from string import punctuation

import spacy
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

# load spacy tokenizer
nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
# df.progress_apply is tqdm method for pandas. It shows progress bar for apply function
# remove the leading and trailing spaces
df['SentimentText'] = df.SentimentText.progress_apply(lambda x: x.strip())

# build vocabulary and corresponding counts
words = Counter()
for sent in tqdm(df.SentimentText.values):
    words.update(w.text.lower() for w in nlp(sent))

# sort with most frequently occuring words first
words = sorted(words, key=words.get, reverse=True)
# add <pad> and <unk> token to vocab which will be used later
words = ['_PAD', '_UNK'] + words

# create word to index dictionary and reverse
word2idx = {o: i for i, o in enumerate(words)}
idx2word = {i: o for i, o in enumerate(words)}


def indexer(s):
    return [word2idx[w.text.lower()] for w in nlp(s)]


# tokenize the tweets and calculate lengths
df['sentimentidx'] = df.SentimentText.progress_apply(indexer)
df['lengths'] = df.sentimentidx.progress_apply(len)

fig = plt.figure(figsize=(8, 5))
ax = sns.distplot(df.lengths.values, kde=False);
ax.set(xlabel='Tweet Length', ylabel='Frequency')
plt.show()