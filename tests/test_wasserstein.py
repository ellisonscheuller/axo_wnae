import torch
from wnae import WNAE


def test_type_preservation():
    model = WNAE(encoder=None, decoder=None)

    n_samples = 100
    n_dim = 10
    samples1 = torch.randn((n_samples, n_dim))
    samples2 = torch.randn((n_samples, n_dim))

    w_dist = model.compute_wasserstein_distance(samples1, samples2)
   
    assert isinstance(w_dist, torch.Tensor)
    assert w_dist.dim() == 0


def test_sanity_checks():
    model = WNAE(encoder=None, decoder=None)

    n_samples = 100
    n_dim = 10
    samples1 = torch.randn((n_samples, n_dim))

    w_dist = model.compute_wasserstein_distance(samples1, samples1)
   
    assert w_dist.item() == 0


def test_gradient_preservation():
    model = WNAE(encoder=None, decoder=None)

    n_samples = 100
    n_dim = 10
    samples1 = torch.randn((n_samples, n_dim), requires_grad=True)
    samples2 = torch.randn((n_samples, n_dim), requires_grad=True)

    w_dist = model.compute_wasserstein_distance(samples1, samples2)

    assert w_dist.requires_grad
