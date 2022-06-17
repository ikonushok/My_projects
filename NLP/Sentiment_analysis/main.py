import pathlib
import re
import warnings
from collections import Counter

import pandas as pd
import spacy

import torch
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
print(df)
print(type(df.SentimentText))
print(f'\nУникальных значений:\n{df.nunique()}')

# split the data into train and validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[['Sentiment']])
train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

print(f'train_df: {train_df.shape},\tval_df: {val_df.shape}')

PAD = 0
UNK = 1


class SentimentDataset(Dataset):
    """Define the pytorch Dataset to process the tweets
       This class can be used for both training and validation dataset
       Run it for training data and pass the word2idx and idx2word when running
       for validation data
    """

    def __init__(self, df, word2idx=None, idx2word=None, max_vocab_size=50000):
        print('Processing Data')
        self.df = df
        print('Removing white space...')
        self.df.SentimentText = self.df.SentimentText.progress_apply(lambda x: str(x).strip())
        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
        if word2idx is None:
            print('Building Counter...')
            word_counter = self.build_counter()
            print('Building Vocab...')
            self.word2idx, self.idx2word = self.build_vocab(word_counter, max_vocab_size)
        else:
            self.word2idx, self.idx2word = word2idx, idx2word
        print('*' * 100)
        print('Dataset info:')
        print(f'Number of Tweets: {self.df.shape[0]}')
        print(f'Vocab Size: {len(self.word2idx)}')
        print('*' * 100)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        sent = self.df.SentimentText[idx]
        tokens = [w.text.lower() for w in self.nlp(self.tweet_clean(sent))]
        vec = self.vectorize(tokens, self.word2idx)
        return vec, self.df.Sentiment[idx]

    def tweet_clean(self, text):
        """Very basic text cleaning. This function can be built upon for
           better preprocessing
        """
        text = re.sub(r'[\s]+', ' ', text)  # replace multiple white spaces with single space
        #         text = re.sub(r'@[A-Za-z0-9]+', ' ', text) # remove @ mentions
        text = re.sub(r'https?:/\/\S+', ' ', text)  # remove links
        text = re.sub(r'[^A-Za-z0-9]+', ' ', text)  # remove non alphanumeric character
        return text.strip()

    def build_counter(self):
        """Tokenize the tweets using spacy and build vocabulary
        """
        words_counter = Counter()
        for sent in tqdm(self.df.SentimentText.values):
            words_counter.update(w.text.lower() for w in self.nlp(self.tweet_clean(sent)))
        return words_counter

    def build_vocab(self, words_counter, max_vocab_size):
        """Add pad and unk tokens and build word2idx and idx2word dictionaries
        """
        word2idx = {'<PAD>': PAD, '<UNK>': UNK}
        word2idx.update(
            {word: i + 2 for i, (word, count) in tqdm(enumerate(words_counter.most_common(max_vocab_size)))})
        idx2word = {idx: word for word, idx in tqdm(word2idx.items())}
        return word2idx, idx2word

    def vectorize(self, tokens, word2idx):
        """Convert tweet to vector
        """
        vec = [word2idx.get(token, UNK) for token in tokens]
        return vec


vocab_size = 100000
train_ds = SentimentDataset(train_df, max_vocab_size=vocab_size)