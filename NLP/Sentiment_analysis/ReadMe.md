
# На основе [BERT для классификации русскоязычных текстов](https://habr.com/ru/post/567028/)
[Git](https://github.com/shitkov/bert4classification)

[RuBERT](https://huggingface.co/cointegrated/rubert-tiny)
___

1.Данные для обучения

Данные очищаются от знаков пунктуации и заглавных букв, после чего, разбиваются случайным образом на три файла (train, val, test)
Вся подготовка ведется в файле [Preprocessing.py](NLP/Sentiment_analysis/Preprocessing.py)

2. Model
3. Helpers
- DataLoader

код:

    from torch.utils.data DataLoader
    train_set = CustomDataset(X_train, y_train, tokenizer)
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

- Optimizer

код:

    from transformers import AdamW
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

- Scheduler

код:

    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=0,
                    num_training_steps=len(train_loader) * epochs
                )

- Loss

код:

    loss_fn = torch.nn.CrossEntropyLoss()

5. Train

Данные в цикле батчами генерируются с помощью DataLoader

    for data in self.train_loader:
        input_ids = data["input_ids"].to(self.device)
        attention_mask = data["attention_mask"].to(self.device)
        targets = data["targets"].to(self.device)

Батч подается в модель:

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask
        )

На выходе получаем распределение вероятности по классам и значение ошибки:

    preds = torch.argmax(outputs.logits, dim=1)
    loss = self.loss_fn(outputs.logits, targets)

Делаем шаг на всех вспомогательных функциях:
- loss.backward(): обратное распространение ошибки; 
- clip_grad_norm(): обрезаем градиенты для предотвращения "взрыва" градиентов; 
- optimizer.step(): шаг оптимизатора; 
- scheduler.step(): шаг планировщика; 
- optimizer.zero_grad(): обнуляем градиенты.

6. Inference

Для предсказания класса для нового текста используется метод predict, который имеет смысл вызывать только после обучения модели. Метод работает следующим образом:
- Токенизируется входной текст; 
- Токенизированный текст подается в модель; 
- На выходе получаем вероятности классов; 
- Возвращаем метку наиболее вероятного класса.

PS:
перепробовал много вариантов..

очень много кодов с недочетами, разбираться с которыми было дольше, чем искать рабочий код, например

[Sentiment Analysis with variable length sequences in Torch](https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130)

[Sentiment Analysis with Variable length sequences in Pytorch](https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130)

[BERT Text Classification Using Pytorch](https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b)

[Code Review: Sentiment Recurrent Neural Networks with PyTorch](https://tracyrenee61.medium.com/code-review-sentiment-recurrent-neural-networks-with-pytorch-fbf5c9624711)

