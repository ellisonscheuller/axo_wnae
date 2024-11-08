import torch

from wnae._mcmc_utils import sample_langevin


def test_mcmc():
    model = lambda x: x[:, 0]**2 + x[:, 1]**2

    n_samples = 100
    x_min = -3
    x_max = 3
    initial_samples = torch.rand((n_samples, 2)) * (x_max - x_min) + x_min

    mcmc_data = sample_langevin(
        initial_samples,
        model,
        n_steps=10,
        step_size=0.02,
        noise_scale=0.2,
        temperature=0.05,
        clip=None,
        clip_grad=None,
        reject_out_of_boundary=False,
        spherical=False,
        mh=False,
    )
    mcmc_samples = mcmc_data["sample"]

    # Check that the MCMC samples have the same shape
    assert mcmc_samples.shape == initial_samples.shape
    
    # Check that MCMC samples are different
    assert torch.any(torch.any(mcmc_samples != initial_samples, dim=1), dim=0)

    x_min = 0
    x_max = 2
    mcmc_data = sample_langevin(
        initial_samples,
        model,
        n_steps=10,
        step_size=0.02,
        noise_scale=0.2,
        temperature=0.05,
        clip=(x_min, x_max),
        clip_grad=None,
        reject_out_of_boundary=False,
        spherical=False,
        mh=False,
    )
    mcmc_samples = mcmc_data["sample"]

    # Check clipping
    assert torch.all(torch.all(mcmc_samples >= x_min, dim=1), dim=0)
    assert torch.all(torch.all(mcmc_samples <= x_max, dim=1), dim=0)

    # Check other options
    mcmc_data = sample_langevin(
        initial_samples,
        model,
        n_steps=10,
        step_size=0.02,
        noise_scale=0.2,
        temperature=0.05,
        clip=(x_min, x_max),
        clip_grad=0.1,
        reject_out_of_boundary=True,
        spherical=True,
        mh=True,
    )
    mcmc_samples = mcmc_data["sample"]

    assert mcmc_samples.shape == initial_samples.shape
