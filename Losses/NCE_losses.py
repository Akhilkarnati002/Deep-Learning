import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchNCELoss(nn.Module):
       """PatchNCE Loss used in the CUT model.

       Computes InfoNCE contrastive loss for multiple layers.
       """

       def __init__(self, temperature: float = 0.07, num_patches: int = 256):
              super().__init__()
              self.temperature = temperature
              self.num_patches = num_patches
              self.cross_entropy_loss = nn.CrossEntropyLoss()

       def forward(self, queries: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
              """
              Args:
                     queries:   Tensor of shape [B, C, HW] — encoded features from generated images
                     positives: Tensor of shape [B, C, HW] — encoded features from real images

              Returns:
                     Scalar contrastive loss (InfoNCE)
              """
              B, C, N = queries.shape  # N = number of patches

              # Select random patch indices (same for queries and positives)
              idx = torch.randperm(N, device=queries.device)[: self.num_patches]
              q = queries[:, :, idx]  # [B, C, num_patches]
              p = positives[:, :, idx]  # [B, C, num_patches]

              # Normalize along channel dimension (cosine similarity)
              q = F.normalize(q, dim=1)
              p = F.normalize(p, dim=1)

              # Compute logits: (num_patches x num_patches) similarity matrix per batch
              logits = torch.bmm(q.permute(0, 2, 1), p) / self.temperature

              # Labels: positive at diagonal for each sample
              labels = torch.arange(self.num_patches, device=queries.device).unsqueeze(0).repeat(B, 1)

              loss = self.cross_entropy_loss(logits.reshape(-1, self.num_patches), labels.reshape(-1))
              return loss
