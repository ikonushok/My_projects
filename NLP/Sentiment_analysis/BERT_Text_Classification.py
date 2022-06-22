# https://habr.com/ru/post/567028/
# https://github.com/shitkov/bert4classification
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

from utilits.bert_classifier import BertClassifier

warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device = ', device)

os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


train_data = pd.read_csv(f'outputs/train.csv')
valid_data = pd.read_csv(f'outputs/valid.csv')
test_data = pd.read_csv(f'outputs/test.csv')


# Initialize BERT classifier
classifier = BertClassifier(

        # model_path='cointegrated/rubert-tiny',
        # tokenizer_path='cointegrated/rubert-tiny',

        model_path='cointegrated/rubert-tiny2-sentence-compression',
        tokenizer_path='cointegrated/rubert-tiny2-sentence-compression',

        # model_path='cointegrated/rubert-tiny-toxicity',  # model fine-tuned for classification of toxicity and inappropriateness for short informal Russian texts, such as comments in social networks
        # tokenizer_path='cointegrated/rubert-tiny-toxicity',

        n_classes=2,
        epochs=10,
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

precision, recall, f1score = precision_recall_fscore_support(labels, predictions,average='macro')[:3]
print(f'precision: {precision}, recall: {recall}, f1score: {f1score}')