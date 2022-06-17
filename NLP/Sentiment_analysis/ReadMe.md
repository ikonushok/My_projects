# [Sentiment Analysis with variable length sequences in Torch](https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130)

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

