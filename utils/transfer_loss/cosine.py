import torch
import torch.nn.functional as F

def cosine_dis(source, target):
    """
    Compute cosine similarity
    source: source domain features (ns * d)
    target: target domain features (nt * d)
    """
    source_mean= torch.mean(source, dim=0, keepdim=True)
    target_mean = torch.mean(target, dim=0, keepdim=True)
    cos_sim = F.cosine_similarity(source_mean, target_mean, dim=1)
    return 1 - cos_sim.mean()