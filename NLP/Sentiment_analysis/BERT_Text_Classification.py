# https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b
# https://colab.research.google.com/drive/1P4Hq0btDUDOTGkCHGzZbAx1lb0bTzMMa?usp=sharing
# https://habr.com/ru/post/567028/
# https://github.com/shitkov/bert4classification

import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from string import punctuation

import torch
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm.auto import tqdm, trange

tqdm.pandas(desc='Progress')
warnings.filterwarnings('ignore')
pd.pandas.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)
pd.set_option("precision", 2)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device = ', device)


train_data = pd.read_csv(f'outputs/train.csv')
valid_data = pd.read_csv(f'outputs/valid.csv')
test_data = pd.read_csv(f'outputs/test.csv')

# from utilits.bert_dataset import CustomDataset
from utilits.bert_classifier import BertClassifier

# Initialize BERT classifier
classifier = BertClassifier(
        model_path='cointegrated/rubert-tiny',
        tokenizer_path='cointegrated/rubert-tiny',
        n_classes=2,
        epochs=2,
        model_save_path='models/bert.pt'
)

# Prepare data and helpers for train and evlauation
classifier.preparation(
        X_train=list(train_data['text']),
        y_train=list(train_data['label']),
        X_valid=list(valid_data['text']),
        y_valid=list(valid_data['label'])
    )

# Train
classifier.train()

# Test
texts = list(test_data['text'])
labels = list(test_data['label'])
predictions = [classifier.predict(t) for t in texts]

from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1score = precision_recall_fscore_support(labels, predictions,average='macro')[:3]
print(f'precision: {precision}, recall: {recall}, f1score: {f1score}')