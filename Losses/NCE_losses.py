import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchNCELoss(nn.Module):
    """PatchNCE Loss used in the CUT model.
    Computes InfoNCE contrastive loss between query features (from generated image)
    and positive/negative key features (from real image).
    """

    
    def __init__(self, temperature: float = 0.07): 
        super().__init__()
        self.temperature = temperature
        # Use CrossEntropyLoss; the "classes" are all keys in the batch.
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: Tensor of shape [B, C, N] — encoded features from generated images G(A).
            keys:    Tensor of shape [B, C, N] — encoded features from real images A or B.

        Returns:
            Scalar InfoNCE loss.
        """
        B, C, N = queries.shape
        
        #Reshape and Normalize (Combine B and N dimensions)
        
        K = B * N
        queries = queries.permute(0, 2, 1).reshape(K, C) # [K, C]
        keys = keys.permute(0, 2, 1).reshape(K, C)       # [K, C]
        
        # Normalize along channel dimension (cosine similarity)
        queries = F.normalize(queries, dim=1)
        keys = F.normalize(keys, dim=1)

        #Compute full similarity matrix Sim (The Logits)
        Sim = torch.matmul(queries, keys.T) / self.temperature
        
        # Target labels: The positive for query q_i is key_i, which lies on the diagonal.
        target = torch.arange(K, device=queries.device)
        
        # Compute Loss
        loss = self.cross_entropy_loss(Sim, target)
        
        return loss