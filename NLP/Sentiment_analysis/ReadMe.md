[Sentiment Analysis with variable length sequences in Torch](https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130)

[Sentiment Analysis with Variable length sequences in Pytorch](https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130)

[BERT Text Classification Using Pytorch](https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b)

# [Code Review: Sentiment Recurrent Neural Networks with PyTorch](https://tracyrenee61.medium.com/code-review-sentiment-recurrent-neural-networks-with-pytorch-fbf5c9624711)

1. Process Data through pytorch Dataset 
- Tokenize tweets
- Build vocabulary
- Vectorize tweets 
2. Make batches through pytorch Dataloader
- Pad tweets to the max length in the batch 
3. Max Pooling and Average Pooling 
- RNN model (GRU) with concat pooling 
4. [Ignite](https://pytorch.org/ignite/) training callbacks 
- Define ignite training loops 
- Add callback for epoch loss and accuracy 
- Add callback for ModelCheckpoint 
- Add callback for EarlyStopping

