
patch = 30
batch_size = 256*4
epochs = 150
alpha = 0.2
num_hard = int(batch_size * 0.5)  # Number of semi-hard triplet examples in the batch
lr = 0.00006
optimiser = 'Adam'
emb_size = 10

path_outputs = 'outputs'
source_root = 'source_root'