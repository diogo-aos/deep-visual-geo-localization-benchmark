import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletMarginLossWithMetricLearning(nn.Module):
    def __init__(self, embedding_size, hidden_dim, margin=1.0, eps=1e-6):
        """
        Args:
        - embedding_size: Dimension of the embedding space.
        - hidden_dim: The hidden dimension, which can differ from embedding_size.
                     Determines the complexity of the learned metric.
        - margin: Margin for the triplet loss.
        - eps: Small constant to avoid numerical instability.
        """
        super(TripletMarginLossWithMetricLearning, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.margin = margin
        self.eps = eps

        # Define the learnable matrix W of shape (hidden_dim, embedding_size)
        self.W = nn.Parameter(torch.Tensor(hidden_dim, embedding_size))

        # Initialize W with Xavier uniform initialization
        nn.init.xavier_uniform_(self.W)

    def compute_distance(self, x1, x2):
        """
        Compute the squared Mahalanobis distance between x1 and x2 using D = W^T W.
        """

        if x1.device != self.W.device:
            self = self.to(x1.device)
        # Compute D = W^T W, which will have shape (embedding_size, embedding_size)
        D = torch.mm(self.W.t(), self.W)  # Shape: (embedding_size, embedding_size)

        # Compute the difference between the embeddings
        diff = x1 - x2  # Shape: (batch_size, embedding_size)

        # Compute the quadratic form diff^T D diff
        # First, multiply diff with D
        # Since the batch dimension is present, we can use bmm for batch matrix multiplication
        # Here diff is a batch of row vectors; we treat each row as a separate instance
        diff_D = torch.matmul(diff, D)  # Shape: (batch_size, embedding_size)

        # Compute element-wise product of diff and diff_D, then sum over embedding dimension
        distance_squared = torch.sum(diff * diff_D, dim=1)  # Shape: (batch_size,)
        return torch.sqrt(distance_squared + self.eps)

    def forward(self, anchor, positive, negative):
        """
        Compute the triplet margin loss using the learned distance metric.
        """

        # Compute distances
        distance_positive = self.compute_distance(anchor, positive)
        distance_negative = self.compute_distance(anchor, negative)

        # Compute triplet loss
        loss = F.relu(distance_positive - distance_negative + self.margin)

        # Return the average loss over the batch
        return loss.mean()