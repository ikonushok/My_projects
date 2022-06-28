
import torch
import torch.nn as nn


class Network(nn.Module):
    """
    Class defines Siamese NN model.
    Attributes
    ----------
    emb_dim: int, optional, default=128.
        Dimension of embedding that input images convert to.
    class_embeddings: nn.Parameter of torch.tensor of shape (10, emb_dim).
        Embeddings corresponding to each class.
    class_embeddings_num: nn.Parameter of torch.tensor of shape (10, ).
        Number of embedding vectors that generate (by summing) embeddings corresponding to each class.
        Need to normalize class_embeddings.
    """

    def __init__(self, emb_dim=128):
        super(Network, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.PReLU(),
            nn.Linear(512, emb_dim)
        )

        self.class_embeddings = nn.Parameter(torch.zeros(10, emb_dim))
        self.class_embeddings_num = nn.Parameter(torch.zeros(10))

    def forward(self, x):
        """
        Convert input images to embeddings.
        Parameters
        ----------
        x: torch.tensor of shape (N, 1, 28, 28).
            Batch of input images.
        Returns
        -------
        torch.tensor of shape (N, emb_dim).
            Batch of embeddings.
        """

        x = self.conv(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        # x = F.normalize(x)
        return x

    def calc_class_embeddings(self, x, labels):
        """
        Computes embeddings corresponding to each class.
        Computes number of embedding vectors that generate (by summing) embeddings corresponding to each class.
        Need to normalize class_embeddings.
        Parameters
        ----------
        x: torch.tensor of shape (N, 1, 28, 28).
            Batch of input images.
        labels: torch.tensor of shape (N, ).
            Labels of input images.
        """

        embeddings = self.forward(x)
        for i in torch.unique(labels):
            self.class_embeddings[i] += embeddings[labels == i].sum(dim=0)
            self.class_embeddings_num[i] += (labels == i).sum()

    def predict(self, x):
        """
        Convert input images to embeddings.
        Parameters
        ----------
        x: torch.tensor of shape (N, 1, 28, 28).
            Batch of input images.
        Returns
        -------
        torch.tensor of shape (N, ).
            Predicted labels.
        """

        embeddings = self.forward(x)
        distances = torch.zeros(x.size(0), self.class_embeddings_num.size(0))
        for i in range(len(embeddings)):
            distances[i] = torch.sqrt((self.class_embeddings / self.class_embeddings_num.unsqueeze(dim=1) -
                            embeddings[i].unsqueeze(dim=0)).pow(2).sum(dim=1))
        return torch.argmin(distances, dim=1)
