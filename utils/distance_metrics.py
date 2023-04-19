import torch


def euclidean_dist(x, y, dim=-1):
    return torch.sqrt(torch.sum((x - y) ** 2, dim=dim))


def cosine_dist(x, y, dim=-1):
    return 1 - torch.nn.functional.cosine_similarity(x, y, dim)


def largest_vals(x, y, dim=-1):
    return - y.sum(dim)