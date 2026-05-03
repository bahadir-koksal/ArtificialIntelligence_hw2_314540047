import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
    """
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        """
        z_i: The projected features of the first augmented view (Batch, 128)
        z_j: The projected features of the second augmented view (Batch, 128)
        """
        batch_size = z_i.size(0)
        
        # Normalize the output vectors (L2 normalization) (because we need direction of feature vecotrs, not magnitude(brightness etc))
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate all views to form the 2N batch: (2*Batch, 128)
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Compute pairwise cosine similarity for the 2N batch
        # Matrix multiplication of representations with its transpose gives cosine similarities 
        # since the vectors are already L2 normalized.
        similarity_matrix = torch.matmul(representations, representations.T)
        
        # Divide by the temperature parameter
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create the "fake labels" to identify the "mate"
        # The mate for image 'k' is at 'k + batch_size' and vice versa.
        labels = torch.cat([torch.arange(batch_size) + batch_size, 
                            torch.arange(batch_size)], dim=0)
        labels = labels.to(z_i.device)
        
        # Mask out the self-similarity (the diagonal of the matrix) so the model doesn't 
        # just match an image with itself (which would be a similarity of 1.0)
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z_i.device)
        similarity_matrix.masked_fill_(mask, -9e15)
        
        # Compute the cross-entropy-style loss
        loss = self.criterion(similarity_matrix, labels)
        
        return loss