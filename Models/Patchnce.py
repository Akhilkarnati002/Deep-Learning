import torch
from torch import nn
from packaging import version

class PatchNCELoss(nn.Module):
    """
    PatchNCE Loss used in CUT / FastCUT.
    Computes contrastive loss between query features (feat_q) and key features (feat_k).
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        # Handle older PyTorch versions for masking
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse("1.2.0") else torch.bool

    def forward(self, feat_q, feat_k):
        """
        Args:
            feat_q: query features of shape [num_patches, dim]
            feat_k: key features of shape [num_patches, dim]
        Returns:
            PatchNCE contrastive loss
        """
        # Ensure both features have the same shape
        assert feat_q.shape == feat_k.shape, "feat_q and feat_k must have the same shape"

        num_patches, dim = feat_q.shape

        # Detach key features to avoid backprop through encoder of the other branch
        feat_k = feat_k.detach()

        # Normalize features for cosine similarity
        feat_q = nn.functional.normalize(feat_q, dim=1)
        feat_k = nn.functional.normalize(feat_k, dim=1)

        # Positive logits: similarity between corresponding patches
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, dim),
            feat_k.view(num_patches, dim, 1)
        ).view(num_patches, 1)

        # Determine batch size for computing negatives
        batch_dim = self.opt.batch_size if not self.opt.nce_includes_all_negatives_from_minibatch else 1
        if num_patches % batch_dim != 0:
            raise ValueError(f"num_patches ({num_patches}) must be divisible by batch_size ({batch_dim})")

        # Reshape features to [batch_size, num_patches_per_image, dim]
        patches_per_image = num_patches // batch_dim
        feat_q = feat_q.view(batch_dim, patches_per_image, dim)
        feat_k = feat_k.view(batch_dim, patches_per_image, dim)

        # Negative logits: similarity between all patches in the same batch/image
        l_neg = torch.bmm(feat_q, feat_k.transpose(2, 1))  # [batch, patches, patches]

        # Mask out diagonal entries (self-similarity)
        diagonal = torch.eye(patches_per_image, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg.masked_fill_(diagonal, -10.0)  # effectively remove positive logit from negatives

        # Flatten negatives
        l_neg = l_neg.view(-1, patches_per_image)

        # Concatenate positive and negative logits and apply temperature
        logits = torch.cat([l_pos, l_neg], dim=1) / self.opt.nce_T

        # Cross-entropy target: first column is positive
        target = torch.zeros(logits.size(0), dtype=torch.long, device=feat_q.device)
        loss = self.cross_entropy_loss(logits, target)

        return loss.mean()  # return mean loss
