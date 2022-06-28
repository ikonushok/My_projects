# https://github.com/Trotts/Siamese-Neural-Network-MNIST-Triplet-Loss/blob/main/Siamese-Neural-Network-MNIST.ipynb
# https://towardsdatascience.com/how-to-train-your-siamese-neural-network-4c6da3259463
"""
Main program with train/validation loops of siamese NN for MNIST dataset classification.
"""

import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
from constants import *
from model import Network
from functions import TripletLoss, MNISTDataset, plot_metric, init_weights

# Fix random parameters for code reproducibility.
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define device: cpu CPU or GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset from torchvision library datasets.
train_dataset = MNIST(root=DATASET_PATH, train=True, download=True, transform=transforms.ToTensor())
# test_dataset = MNIST(root=dataset_path, train=False, download=True, transform=transforms.ToTensor())

# Split dataset into train and validation datasets.
# Create tensors of images of shape (N, 1, 28, 28) and labels of shape (N, ).
train_imgs = torch.vstack(list(train_dataset[i][0].unsqueeze(0) for i in range(0, len(train_dataset), 60)))  # 1000, 28, 28
train_labels = torch.tensor(list(train_dataset[i][1] for i in range(0, len(train_dataset), 60)))  # 1000
val_imgs = torch.vstack(list(train_dataset[i][0].unsqueeze(0) for i in range(1, len(train_dataset), 60)))  # 1000, 28, 28
val_labels = torch.tensor(list(train_dataset[i][1] for i in range(1, len(train_dataset), 60)))  # 1000

# Create train and validation instances of MNISTDataset and corresponding Dataloaders.
train_dataset = MNISTDataset(train_imgs, train_labels, is_train=True, transform=None)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_dataset = MNISTDataset(val_imgs, val_labels, is_train=True, transform=None)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# Define model.
model = Network(EMBEDDING_DIMS)
# Initialize Conv2d layers according to He normal distribution.
model.apply(init_weights)
model = model.to(device)
# Define optimizer and loss function.
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = TripletLoss()

# Define dictionary for loss function values at each train/validation epoch.
losses = {'train': [], 'val': []}
for epoch in range(EPOCHS):
    train_loss = 0.
    model.train()
    # Train model.
    for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(train_loader):
        optimizer.zero_grad()
        anchor_out = model(anchor_img.to(device))
        positive_out = model(positive_img.to(device))
        negative_out = model(negative_img.to(device))
        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    losses['train'].append(train_loss / BATCH_SIZE / len(train_loader))
    # Validate model.
    val_loss = 0.
    model.eval()
    with torch.no_grad():
        for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(val_loader):
            anchor_out = model(anchor_img.to(device))
            positive_out = model(positive_img.to(device))
            negative_out = model(negative_img.to(device))
            loss = criterion(anchor_out, positive_out, negative_out)
            val_loss += loss.item()
    losses['val'].append(val_loss / BATCH_SIZE / len(val_loader))

    print("Epoch {} Train_loss {:.4f} Val_loss {:.4f}".format(epoch, losses['train'][-1], losses['val'][-1]))

# Plot loss function values for train/validation.
plot_metric(losses, title='Loss')

# Define list of embeddings of train images.
train_embeddings = []
# Define list of corresponding labels.
labels = []
model.eval()
with torch.no_grad():
    for img, _, _, label in tqdm(train_loader, desc='Create class embeddings'):
        # Evaluate embeddings of train images.
        train_embeddings.append(model(img.to(device)).cpu().numpy())
        labels.append(label)

        # Evaluate embeddings corresponding to each class and number
        # of embedding vectors that generate (by summing) embeddings corresponding to each class.
        model.calc_class_embeddings(img.to(device), label.to(device))
    # Calculate normalized embeddings of each class.
    class_embeddings = model.class_embeddings.cpu().numpy() / np.expand_dims(
                                                                model.class_embeddings_num.cpu().numpy(), 1)
train_results = np.concatenate(train_embeddings)
labels = np.concatenate(labels)

# Plot embeddings of train images and embeddings of each class (see plot).
plt.figure(figsize=(15, 10), facecolor="azure")
for label in np.unique(labels):
    tmp = train_results[labels == label]
    plt.scatter(tmp[:, 0], tmp[:, 1], label=label)
plt.scatter(class_embeddings[:, 0], class_embeddings[:, 1], c='black', s=200, marker='>')
plt.legend()
plt.show()

# Evaluate accuracy of classification using validation dataset.
accuracy = 0.
model.eval()
with torch.no_grad():
    for img, _, _, label in tqdm(val_loader, desc='Predict'):
        accuracy += (model.predict(img.to(device)) == label).sum().item()
print('Val_accuracy:', round(accuracy / BATCH_SIZE / len(val_loader) * 100., 1), '%')

# Take an example of each class and calculate its embedding.
model.eval()
with torch.no_grad():
    random_imgs = []
    labels = np.unique(val_labels)
    for label in labels:
        img = random.choice(val_imgs[val_labels == label])
        random_imgs.append(img.reshape(1, 1, 28, 28))
    random_imgs = torch.vstack(random_imgs)
    random_embeddings = model(random_imgs.to(device)).cpu().numpy()

# Plot an embedding of each class and embeddings of classes to look at their proximity.
plt.figure(figsize=(15, 10))
colors = cm.rainbow(np.linspace(0, 1, len(labels)))
for label in labels:
    plt.scatter(random_embeddings[label, 0], random_embeddings[label, 1], label=label, color=colors[label])
plt.scatter(class_embeddings[:, 0], class_embeddings[:, 1], s=300, marker='>', color=colors)
plt.legend()
plt.show()

# Evaluate distances between anchors and positives, anchors and negatives
dist = {'positive': [], 'negative': []}
model.eval()
with torch.no_grad():
    for anchor_img, positive_img, negative_img, label in train_loader:
        anchor_emb = model(anchor_img.to(device)).cpu()
        positive_emb = model(positive_img.to(device)).cpu()
        negative_emb = model(negative_img.to(device)).cpu()
        dist['positive'].append(torch.sqrt((anchor_emb - positive_emb).pow(2).sum(dim=1)))
        dist['negative'].append(torch.sqrt((anchor_emb - negative_emb).pow(2).sum(dim=1)))
    dist['positive'] = torch.hstack(dist['positive'])
    dist['negative'] = torch.hstack(dist['negative'])

# Plot distance distribution
ax = sns.displot(dist, kde=True, stat='density')
ax.set(xlabel='Pairwise distance')
plt.show()
