import os
import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from NLP.Sentiment_analysis.constants import first_n_words, lr, batch_size, destination_folder


class CustomDataset(Dataset):

  def __init__(self, texts, targets, tokenizer, max_len=first_n_words):
    self.texts = texts
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = str(self.texts[idx])
    target = self.targets[idx]

    encoding = self.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    return {
      'text': text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }


class BertClassifier():

    os.environ['PYTHONHASHSEED'] = str(42)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def __init__(self, max_len, n_classes, lr, batch_size, epochs,
                 model_path, tokenizer_path, model_save_path='models/bert.pt'):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_save_path = model_save_path
        self.max_len = max_len
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.out_features = self.model.bert.encoder.layer[1].output.dense.out_features
        self.model.classifier = torch.nn.Linear(self.out_features, n_classes)
        self.model.to(self.device)

    def preparation(self, X_train, y_train, X_valid, y_valid):
        # create datasets
        self.train_set = CustomDataset(X_train, y_train, self.tokenizer)
        self.valid_set = CustomDataset(X_valid, y_valid, self.tokenizer)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=batch_size, shuffle=True)

        # helpers initialization
        self.optimizer = AdamW(self.model.parameters(), lr=lr, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    def fit(self):
        self.model = self.model.train()
        losses = []
        correct_predictions = 0

        for data in self.train_loader:
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            targets = data["targets"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)
            loss = self.loss_fn(outputs.logits, targets)

            correct_predictions += torch.sum(preds == targets)

            losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        train_acc = correct_predictions.double() / len(self.train_set)
        train_loss = np.mean(losses)
        return train_acc, train_loss

    def eval(self):
        self.model = self.model.eval()
        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for data in self.valid_loader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                targets = data["targets"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=1)
                loss = self.loss_fn(outputs.logits, targets)
                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())

        val_acc = correct_predictions.double() / len(self.valid_set)
        val_loss = np.mean(losses)
        return val_acc, val_loss

    def train(self, modelname):
        best_accuracy = 0
        arr_train_loss, arr_train_acc, arr_val_loss, arr_val_acc = [], [], [], []
        for epoch in range(self.epochs):
            print(f'{"-"*10}\nEpoch {epoch + 1}/{self.epochs}')
            train_acc, train_loss = self.fit()
            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = self.eval()
            print(f'Val loss {val_loss} accuracy {val_acc}')

            arr_train_loss.append(train_loss)
            arr_train_acc.append(train_acc)
            arr_val_loss.append(val_loss)
            arr_val_acc.append(val_acc)

            if val_acc > best_accuracy:
                torch.save(self.model, self.model_save_path)
                print('Save best model..')
                best_accuracy = val_acc

            plt.figure(figsize=(6, 4))
            plt.plot(arr_train_acc, label='train_accuracy')
            plt.plot(arr_val_acc, label='val_accuracy')
            plt.plot(arr_train_loss, label='train_CrossEntropyLoss')
            plt.plot(arr_val_loss, label='val_CrossEntropyLoss')
            plt.legend()
            plt.grid()
            plt.title(f'history_{modelname}:\n'
                      f'for {epoch+1} epochs  (first_n_words={first_n_words}, lr={lr}, batch_size={batch_size})')
            plt.savefig(f'outputs/{modelname}.png')
            plt.show()

            # Сохраним статистику в файл
            df_test_metrics = pd.DataFrame(
                list(zip(np.asarray(arr_train_acc), np.asarray(arr_val_acc), arr_train_loss, arr_val_loss)),
                columns=['train_accuracy', 'val_accuracy', 'train_CrossEntropyLoss', 'val_CrossEntropyLoss'])
            df_test_metrics.to_csv(f'{destination_folder}/history_{modelname}.csv')

        self.model = torch.load(self.model_save_path)

    def predict(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        out = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        input_ids = out["input_ids"].to(self.device)
        attention_mask = out["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )

        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

        return prediction