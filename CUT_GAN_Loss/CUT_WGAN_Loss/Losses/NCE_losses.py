import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchNCELoss(nn.Module):
    """PatchNCE Loss used in the CUT model.
    Computes InfoNCE contrastive loss between query features (from generated image)
    and positive/negative key features (from real image).
    """

    # Note: We do NOT need num_patches in the constructor anymore.
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
        
        # 1. Reshape and Normalize (Combine B and N dimensions)
        # Total number of queries/keys in the batch: K = B * N
        K = B * N
        queries = queries.permute(0, 2, 1).reshape(K, C) # [K, C]
        keys = keys.permute(0, 2, 1).reshape(K, C)       # [K, C]
        
        # Normalize along channel dimension (cosine similarity)
        queries = F.normalize(queries, dim=1)
        keys = F.normalize(keys, dim=1)

        # 2. Compute full similarity matrix Sim (The Logits)
        # Sim: [K, K] where K=B*N. 
        # Query i is compared against ALL K keys (the K-1 negatives and 1 positive).
        Sim = torch.matmul(queries, keys.T) / self.temperature
        
        # 3. Target labels: The positive for query q_i is key_i, which lies on the diagonal.
        # The target index for query i is simply i (0, 1, 2, ..., K - 1)
        # K is the correct size for both the input (Sim) batch_size and the target labels.
        target = torch.arange(K, device=queries.device)
        
        # 4. Compute Loss
        # Sim: logits (K x K), target: true class indices (K)
        # If B=1 and N=256, then K=256. Sim is [256, 256]. Target is [256].
        # This resolves the error.
        loss = self.cross_entropy_loss(Sim, target)
        
        return loss