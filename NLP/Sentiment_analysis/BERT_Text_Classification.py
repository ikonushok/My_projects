# https://habr.com/ru/post/567028/
# https://github.com/shitkov/bert4classification
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support

from utilits.bert_classifier import BertClassifier
from constants import first_n_words, source_root, lr, batch_size, epochs, destination_folder

warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device = ', device)

os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.mkdir('outputs')


train_data = pd.read_csv(f'{source_root}/train.csv')
valid_data = pd.read_csv(f'{source_root}/valid.csv')
test_data = pd.read_csv(f'{source_root}/test.csv')


model_path = ['rubert-tiny-toxicity', 'rubert-tiny2-sentence-compression', 'rubert-tiny']
arr_precision, arr_recall, arr_f1score = [], [], []
for modelname in model_path:
        # Initialize BERT classifier
        classifier = BertClassifier(

                max_len=first_n_words,
                n_classes=2,
                lr=lr,
                batch_size=batch_size,
                epochs=epochs,

                model_path=f'cointegrated/{modelname}',
                tokenizer_path=f'cointegrated/{modelname}',

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
        classifier.train(modelname)


        # Test
        texts = list(test_data['text'])
        labels = list(test_data['label'])
        predictions = [classifier.predict(t) for t in texts]

        precision, recall, f1score = precision_recall_fscore_support(labels, predictions,average='macro')[:3]
        arr_precision.append(precision)
        arr_recall.append(recall)
        arr_f1score.append(f1score)
        print(f'precision: {precision}, recall: {recall}, f1score: {f1score}')


# Сохраним статистику в файл
df_test_metrics = pd.DataFrame(list(zip(model_path, arr_precision, arr_recall, arr_f1score)),
                               columns=['model', 'precision', 'arr_recall', 'f1score'])
df_test_metrics = df_test_metrics.set_index('model')
print(df_test_metrics)
df_test_metrics.to_excel(f'{destination_folder}/test_metrics.xlsx')
