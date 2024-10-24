import numpy as np
import torch
import torch.autograd as autograd


def langevin_step(
        x,
        energy_x,
        grad_energy_x,
        model,
        step_size,
        noise_scale,
        temperature,
        clip,
        clip_grad,
        reject_boundary,
        spherical,
        mh,
    ):
    """Step in Metropolis-adjusted Langevin Monte Carlo algorithm."""

    results_dict = {}

    # Apply step
    noise = torch.randn_like(x) * noise_scale
    drift = - step_size * grad_energy_x
    step = drift / temperature + noise
    y = x + step

    # Clip gradient and boundary rejection
    reject = torch.zeros(len(y), dtype=torch.bool)
    if clip is not None:
        if reject_boundary:
            accept = ((y >= clip[0]) & (y <= clip[1])).view(len(x), -1).all(dim=1)
            reject = ~ accept
            y[reject] = x[reject]
        else:
            y = torch.clamp(y, clip[0], clip[1])

    # Projection on hyper-sphere
    if spherical:
        y = y / y.norm(dim=1, p=2, keepdim=True)

    # Compute gradient after step
    energy_y = model(y)
    grad_energy_y = autograd.grad(energy_y.sum(), y, create_graph=True)[0]
 
    # Clip gradient
    if clip_grad is not None:
        grad_energy_y = torch.clamp(grad_energy_y, -clip_grad, clip_grad)

    # Metropolis-Hasting rejection
    if mh:
        y_to_x = ((grad_energy_x + grad_energy_y) * step_size - noise).view(len(x), -1).norm(p=2, dim=1, keepdim=True) ** 2
        x_to_y = (noise).view(len(x), -1).norm(dim=1, keepdim=True, p=2) ** 2
        transition = - (y_to_x - x_to_y) / 4 / step_size  # B x 1
        prob = -energy_y + energy_x
        accept_prob = torch.exp((transition + prob) / temperature)[:,0]  # B
        reject = (torch.rand_like(accept_prob) > accept_prob)
        y[reject] = x[reject]
        energy_y[reject] = energy_x[reject]
        grad_energy_y[reject] = grad_energy_x[reject]

    # Metrics to monitor
    results_dict.update({
        "drift": drift.detach().cpu(),
        "diffusion": noise.detach().cpu(),
        "steps": step.detach().cpu(),
    })

    return y, energy_y, grad_energy_y, results_dict


def sample_langevin(
        x,
        model,
        n_steps,
        step_size=None,
        noise_scale=None,
        temperature=1.,
        clip=None,
        clip_grad=None,
        reject_boundary=False,
        spherical=False,
        mh=False,
    ):
    """Langevin Monte Carlo (with Metroplis-Hasting algorithm if mh is True).

    Args:
        x (torch.Tensor): Initial points from which to start the MCMC.
        model (any): An energy-based model returning energy.
        n_steps (integer): Number of MCMC stpe to run.
        step_size (float or None): If None, set to `noise_scale`**2 / 2.
        noise_scale (float or None): If None, set to np.sqrt(`step_size` * 2).
        clip (tuple or None): If None samples are not clipped, otherwise
            samples are clipped in the provided boundaries.
        clip_grad (float or None): If not None, clip gradient to the given value.
        reject_boundary (bool): Reject out-of-domain samples if True. Otherwise clip.
        spherical (bool): Is True, project onto the hyper-shere of unit radius.
        mh (bool): Use Metropolis-Hastings rejection.
        temperature (float): Divide energy by temperature, can be seen as
            changing step size.
    
    Returns:
        dict
    """

    if noise_scale is None:
        noise_scale = np.sqrt(2 * step_size)
    if step_size is None:
        step_size = (noise_scale ** 2) / 2

    x.requires_grad = True

    # Book lists to monitore evolution of the MCMC
    sampler_dict = {}
    sampler_dict["samples"] = []

    # Compute gradient
    energy_x = model(x)
    grad_energy_x = autograd.grad(energy_x.sum(), x, create_graph=True)[0]
    if clip_grad is not None:
        grad_energy_x = torch.clamp(grad_energy_x, -clip_grad, clip_grad)

    # Run the chain
    sampler_dict["samples"].append(x.detach().cpu())
    for i_step in range(n_steps):
        # Pass and return energy and gradient as well as data points to speed up the MCMC
        y, energy_y, grad_energy_y, results_dict = langevin_step(
            x,
            energy_x,
            grad_energy_x,
            model,
            step_size,
            noise_scale,
            temperature,
            clip,
            clip_grad,
            reject_boundary,
            spherical,
            mh,
        )

        # Update the point, energy and gradient
        x = y
        energy_x = energy_y
        grad_energy_x = grad_energy_y

        # Update results dict
        sampler_dict["samples"].append(x.detach().cpu())
        for key, value in results_dict.items():
            if key not in sampler_dict.keys():
                sampler_dict[key] = [value]
            else:
                sampler_dict[key].append(value)
    
    sampler_dict["sample"] = x

    return sampler_dict
