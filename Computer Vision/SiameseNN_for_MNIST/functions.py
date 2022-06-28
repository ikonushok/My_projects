
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def plot_metric(loss, title='Loss'):
    """
    Plot metrics of NN train/val process.
    Parameters
    ----------
    loss: dict, keys = 'train', 'val', values = lists.
        Dictionary contains train and validation metric values at corresponding lists.
    title: str, optional, default='Loss'.
        Graph title, y-axis label.
    Returns
    -------
        Graph with train and validation plots.
    """

    plt.plot(loss['train'])
    plt.plot(loss['val'])
    plt.title(title)
    plt.ylabel(title)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.grid()
    plt.show()


def init_weights(m):
    """
    Normal He initialization of convolutional Conv2d layers of NN model.
    Parameters
    ----------
    m: torch.nn.Module.
        Submodule of NN class.
    Returns
    -------
    torch.nn.Module.
        Normal He initialized submodule (Conv2d layer) of NN model.
    """

    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


class TripletLoss(nn.Module):
    """
    Class used to calculate triplet loss.
    Attributes
    ----------
    margin: float, optional, default=1.0.
        Margin between anchor-positive and anchor-negative distances.
    """

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    @staticmethod
    def calc_euclidean(x1, x2):
        """
        Calculates distances between embeddings containing in two sets (batches).
        Parameters
        ----------
        x1: torch.tensor of shape (N, embedding_dims).
            A set of embeddings.
        x2: torch.tensor of shape (N, embedding_dims).
            The other set of embeddings.
        Returns
        -------
        torch.tensor of shape (N, ).
            Distances between embeddings.
        """

        return torch.sqrt((x1 - x2).pow(2).sum(dim=1))

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Calculates triplet loss value.
        Parameters
        ----------
        anchor: torch.tensor of shape (N, embedding_dims).
            A set of anchor class embeddings.
        positive: torch.tensor of shape (N, embedding_dims).
            A set of positive class embeddings.
        negative: torch.tensor of shape (N, embedding_dims).
            A set of negative class embeddings.
        Returns
        -------
        torch.tensor of shape (1, ).
            Triplet loss value.
        """

        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


class MNISTDataset(Dataset):
    """
    Class used to generate triplets from MNIST dataset.
    Attributes
    ----------
    is_train: bool, optional, default=True.
        If True, then generates triplets and labels to train/validate model.
        If False, then generates single images to test model (to make prediction).
    transform: optional, default=None.
        An instance of torchvision Compose class containing augmenting functions.
    imgs: torch.tensor of shape (N, 1, 28, 28).
        A tensor containing MNIST images.
    labels: torch.tensor of shape (N, ).
        Labels of corresponding MNIST images.
        If is_train=False, then labels may be not assigned.
    indices: torch.tensor of shape (N, ).
        Indices of image-label pairs.
    """

    def __init__(self, imgs, labels=None, is_train=True, transform=None):
        self.is_train = is_train
        self.transform = transform

        if self.is_train:
            self.imgs = imgs
            self.labels = labels
            self.indices = torch.tensor(range(len(labels)))
        else:
            self.imgs = imgs

    def __len__(self):
        """
        Measures object's length (dataset size).
        Returns
        -------
        int.
            Dataset size.
        """

        return len(self.labels)

    def __getitem__(self, item):
        """
        Opens, augments, preprocesses and returns idx-th tuple of image and mask.
        Parameters
        ----------
        item: int.
            Index number of image and corresponding label.

        Returns
        -------
        If is_train=True, then Tuple of length 4.
        If is_train=False, then torch.tensor of shape (1, 28, 28).
            Triplet of augmented anchor class, positive class and negative class images and anchor label.
        """

        anchor_img = self.imgs[item].reshape(1, 28, 28)

        if self.is_train:
            anchor_label = self.labels[item]
            positive_indices = self.indices[self.indices != item][self.labels[self.indices != item] == anchor_label]

            positive_item = random.choice(positive_indices)
            positive_img = self.imgs[positive_item].reshape(1, 28, 28)

            negative_indices = self.indices[self.indices != item][self.labels[self.indices != item] != anchor_label]
            negative_item = random.choice(negative_indices)
            negative_img = self.imgs[negative_item].reshape(1, 28, 28)

            if self.transform:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)

            return anchor_img, positive_img, negative_img, anchor_label

        else:
            if self.transform:
                anchor_img = self.transform(anchor_img)
            return anchor_img
