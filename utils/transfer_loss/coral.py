import torch


def CORAL(source, target):
    """
    Compute CORAL loss
    source: source domain features (ns * d)
    target: target domain features (nt * d)
    """
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    source_mean = torch.mean(source, 0, keepdim=True)
    target_mean = torch.mean(target, 0, keepdim=True)
    source = source - source_mean
    target = target - target_mean

    cs = torch.mm(source.t(), source) / (ns-1)
    ct = torch.mm(target.t(), target) / (nt-1)

    loss = torch.norm(cs - ct, p='fro') ** 2 / (4 * d * d)
    return loss
