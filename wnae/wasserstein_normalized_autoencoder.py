import numpy as np
import torch
import ot

from wnae._sample_buffer import SampleBuffer
from wnae._mcmc_utils import sample_langevin
from wnae._logger import log


def mse(x, reco):
    n_dim = np.prod(x.shape[1:])
    return ((x - reco) ** 2).view((x.shape[0], -1)).sum(dim=1) / n_dim


class WNAE(torch.nn.Module):
    """Wasserstein Normalized Autoencoder class.

    Args:
        encoder (torch.nn.Module): Encoder network.
        decoder (torch.nn.Module): Decoder network.
        sampling (str): Sampling methods, choose from 'cd' (Contrastive
            Divergence), 'pcd' (Persistent CD), 'omi' (on-manifold
            initialization).
        x_step (int): Number of steps in MCMC.
        x_step_size (float or None): Step size of the MCMC. If None, will be set
            to `x_noise_std**2 / 2`.
        x_noise_std (float or None): Noise standard deviation in the MCMC.
            If None, will be set to np.sqrt(`2 * step_size`).
        x_temperature (float): Temperature of the MCMC.
        x_bound (tuple or None): Min and max values to clip the MCMC samples.
            If None, samples will not be clipped.
        x_clip_grad (tuple or None): Min and max values to clip the gradient
            norm. If None, gradient will not be clipped.
        x_mh (bool): If True, use Metropolis-Hastings rejection in MCMC.
        z_step (int): Same as `x_step` but for the latent MCMC.
        z_step_size (float): Same as `x_step_size` but for the latent MCMC.
        z_noise_std (float): Same as `x_noise_std` but for the latent MCMC.
        z_temperature (float): Same as `x_temperature` but for the latent MCMC.
        z_bound (tuple or None): Same as `x_bound` but for the latent MCMC.
        z_clip_grad (tuple or None): Same as `x_clip_grad` but for the latent MCMC.
        z_mh (bool): Same as `x_mh` but for the latent MCMC.
        reco_error_function (callable, optional): the reconstruction error between
            two tensors. Default is the mean squared error (MSE).
        spherical (bool): Project latent vectors onto the hypersphere.
        initial_dist (str): Distribution from which initial samples are
            generated. Choose from 'gaussian' or 'uniform'.
        replay (bool): Whether to use the replay buffer.
        replay_ratio (float): For PCD, probability to keep sample for
            the initialization of the next chain.
        buffer_size (int): Size of replay buffer.
    """

    def __init__(
        self,
        encoder,
        decoder,
        sampling="pcd",
        x_step=50,
        x_step_size=10,
        x_noise_std=0.05,
        x_temperature=1.0,
        x_bound=(0, 1),
        x_clip_grad=None,
        x_reject_boundary=False,
        x_mh=False,
        z_step=50,
        z_step_size=0.2,
        z_noise_std=0.2,
        z_temperature=1.0,
        z_bound=None,
        z_clip_grad=None,
        z_reject_boundary=False,
        reco_error=mse,
        z_mh=False,
        spherical=False,
        initial_dist="gaussian",
        replay=True,
        replay_ratio=0.95,
        buffer_size=10000,
    ):
        # Building on top of https://github.com/swyoon/normalized-autoencoders
        super().__init__()

        x_step_size, x_noise_std = self.__mcmc_checks_and_definition(x_step_size, x_noise_std)
        z_step_size, z_noise_std = self.__mcmc_checks_and_definition(z_step_size, z_noise_std)

        self.encoder = encoder
        self.decoder = decoder
        self.sampling = sampling
        
        self.x_step = x_step
        self.x_step_size = x_step_size
        self.x_noise_std = x_noise_std
        self.x_temperature = x_temperature
        self.x_bound = x_bound
        self.x_clip_grad = x_clip_grad
        self.x_mh = x_mh
        self.x_reject_boundary = x_reject_boundary
        self.x_shape = None

        self.z_step = z_step
        self.z_step_size = z_step_size
        self.z_noise_std = z_noise_std
        self.z_temperature = z_temperature
        self.z_bound = z_bound
        self.z_clip_grad = z_clip_grad
        self.z_mh = z_mh
        self.z_reject_boundary = z_reject_boundary
        self.z_shape = None

        self.error = reco_error

        self.spherical = spherical
        self.initial_dist = initial_dist
        self.replay = replay
        self.replay_ratio = replay_ratio
        self.buffer_size = buffer_size
        self.buffer = SampleBuffer(max_samples=buffer_size, replay_ratio=replay_ratio)

    @property
    def __sample_shape(self):
        if self.sampling == "omi":
            return self.z_shape
        else:
            return self.x_shape

    @staticmethod
    def __mcmc_checks_and_definition(step_size, noise_std):
        assert step_size is not None or noise_std is not None
        assert step_size is None or step_size > 0

        if step_size is None or noise_std is None:
            if step_size is None:
                step_size = noise_std**2 / 2.
            else:
                noise_std = np.sqrt(2 * step_size)

        return step_size, noise_std

    @staticmethod
    def compute_wasserstein_distance(positive_samples, negative_samples):
        """Compute the Wasserstein distance between two sets of samples.

        Uses `ot.emd2 <https://pythonot.github.io/all.html#ot.emd2>`__.

        Args:
            positive_samples (torch.Tensor)
            negative_samples (torch.Tensor)
        
        Returns:
            float
        """

        if int(ot.__version__.split(".")[0]) == 0 and int(ot.__version__.split(".")[1]) < 9:
            log.warning(f"Your optimal transport ot version is {ot.__version__}")
            log.warning(f"EMD calculation not supported for gradient descent.")
            exit(1)
        loss_matrix = ot.dist(positive_samples, negative_samples)
        n_examples = len(positive_samples)
        weights = torch.ones(n_examples) / n_examples
        emd = ot.emd2(
            weights,
            weights,
            loss_matrix,
            numItermax=1e6,
        )

        return emd

    @staticmethod
    def __project_onto_hypersphere(z):
        """Project onto the hypersphere of unit length."""

        if len(z.shape) == 4:
            z = z / z.view(len(z), -1).norm(dim=-1)[:, None, None, None]
        else:
            z = z / z.view(len(z), -1).norm(dim=1, keepdim=True)

    def __encode(self, x):
        if self.spherical:
            return self.__project_onto_hypersphere(self.encoder(x))
        else:
            return self.encoder(x)

    def forward(self, x):

        z = self.__encode(x)
        recon = self.decoder(z)
        return self.error(x, recon)
    
    def __energy(self, x):
        return self.forward(x)

    def __energy_with_z(self, x):
        z = self.__encode(x)
        recon = self.decoder(z)
        return self.error(x, recon), z

    def __set_x_shape(self, x):
        if self.x_shape is None:
            self.x_shape = x.shape[1:]

    def __set_z_shape(self, x):
        if self.z_shape is None:
            # infer z_shape by computing forward
            with torch.no_grad():
                dummy_z = self.__encode(x[[0]])
            z_shape = dummy_z.shape
            self.z_shape = z_shape[1:]

    def __set_shapes(self, x):
        self.__set_z_shape(x)
        self.__set_x_shape(x)

    def __initial_sample(self, n_samples, device):
        l_sample = []
        if not self.replay or len(self.buffer) == 0:
            n_replay = 0
        else:
            n_replay = (np.random.rand(n_samples) < self.replay_ratio).sum()
            l_sample.append(self.buffer.get(n_replay))

        shape = (n_samples - n_replay,) + self.__sample_shape
        if self.initial_dist == "gaussian":
            x0_new = torch.randn(shape, dtype=torch.float)
        elif self.initial_dist == "uniform":
            x0_new = torch.rand(shape, dtype=torch.float)
            if self.sampling != "omi" and self.x_bound is not None:
                x0_new = x0_new * (self.x_bound[1] - self.x_bound[0]) + self.x_bound[0]
            elif self.sampling == "omi" and self.z_bound is not None:
                x0_new = x0_new * (self.z_bound[1] - self.z_bound[0]) + self.z_bound[0]
        else:
            log.critical(f"Invalid initial distribution {self.initial_dist}")
            exit(1)

        l_sample.append(x0_new)
        return torch.cat(l_sample).to(device)

    def __sample_x(self, n_sample=None, device=None, x0=None, replay=False):
        if x0 is None:
            x0 = self.__initial_sample(n_sample, device=device)

        mcmc_data = sample_langevin(
            x0.detach(),
            self.__energy,
            n_steps=self.x_step,
            step_size=self.x_step_size,
            noise_scale=self.x_noise_std,
            temperature=self.x_temperature,
            clip=self.x_bound,
            clip_grad=self.x_clip_grad,
            spherical=False,
            mh=self.x_mh,
            reject_boundary=self.x_reject_boundary,
        )

        mcmc_data["sample_x"] = mcmc_data.pop("sample")
        if replay:
            self.buffer.push(mcmc_data["sample_x"])

        return mcmc_data

    def __sample_z(self, n_sample=None, device=None, replay=False, z0=None):
        if z0 is None:
            z0 = self.__initial_sample(n_sample, device)
        energy = lambda z: self.__energy(self.decoder(z))
        mcmc_data = sample_langevin(
            z0,
            energy,
            step_size=self.z_step_size,
            n_steps=self.z_step,
            noise_scale=self.z_noise_std,
            temperature=self.temperature,
            clip=self.z_bound,
            clip_grad=self.z_clip_langevin_grad,
            spherical=self.spherical,
            mh=self.z_mh,
            reject_boundary=self.z_reject_boundary,
        )

        if replay:
            self.buffer.push(mcmc_data["sample"])
        return mcmc_data

    def __sample_omi(self, n_sample, device, replay=False):
        """Sample using on-manifold initialization."""

        mcmc_data = {}

        # Step 1: On-manifold initialization: LMC on Z space
        z0 = self.__initial_sample(n_sample, device)
        if self.spherical:
            z0 = self.__project_onto_hypersphere(z0)

        mcmc_data_z = self.__sample_z(z0=z0, replay=replay)
        sample_z = mcmc_data_z.pop("sample")
        mcmc_data.update({
            f"{k}_z": v for k, v in mcmc_data_z.items()
        })

        sample_x_1 = self.decoder(sample_z).detach()
        if self.x_bound is not None:
            sample_x_1.clamp_(self.x_bound[0], self.x_bound[1])

        # Step 2: LMC on X space
        mcmc_data_x = self.__sample_x(x0=sample_x_1, replay=False)
        mcmc_data.update(mcmc_data_x)

        return mcmc_data

    def __sample(self, x0=None, n_sample=None, device=None, replay=None):
        """Sampling factory function.
        
        Takes either x0 or n_sample and device.
        """

        if x0 is not None:
            n_sample = len(x0)
            device = x0.device
        if replay is None:
            replay = self.replay

        if self.sampling == "cd":
            return self.__sample_x(n_sample, device, x0=x0, replay=False)
        elif self.sampling == "pcd":
            return self.__sample_x(n_sample, device, replay=replay)
        elif self.sampling == "omi":
            return self.__sample_omi(n_sample, device, replay=replay)

    def __wnae_step(self, x, mcmc_replay=True, compute_emd=True, detach_negative_samples=False, run_mcmc=True):
        """WNAE step.
        
        Args:
            x (torch.Tensor): Data
            mcmc_replay (bool, optional, default=True): Set to True if the MCMC
                samples obtained should be added to the buffer for replay
        """

        self.__set_shapes(x)
        positive_energy, positive_z = self.__energy_with_z(x)
        ae_loss = positive_energy.mean()

        training_dict = {
            "reco_errors": positive_energy.detach().cpu(),
            "positive_energy": positive_energy.mean().item(),
            "positive_z": positive_z.detach().cpu(),
        }

        if run_mcmc:
            mcmc_data = self.__sample(x, replay=mcmc_replay)
            negative_samples = mcmc_data.pop("sample_x")
            if detach_negative_samples:
                negative_samples = negative_samples.detach()

            negative_energy, negative_z = self.__energy_with_z(negative_samples)

            if compute_emd:
                loss = self.compute_wasserstein_distance(x, negative_samples)
                training_dict["loss"] = loss.item()
            else:
                loss = None

            nae_loss = positive_energy.mean() - negative_energy.mean()

            training_dict.update({
                "negative_samples": negative_samples.detach().cpu(),
                "negative_energy": negative_energy.mean().item(),
                "negative_z": negative_z.detach().cpu(),
                "mcmc_data": mcmc_data,
            })

        else:
            loss = None
            nae_loss = None

        return loss, ae_loss, nae_loss, training_dict

    def train_step(self, x):
        """WNAE training step.
        
        Returns the WNAE loss and information about the training on the provided data.
        Use as:
        
        .. code-block:: python

            model.train()
            optimizer.zero_grad()
            loss, training_dict = model.train_step(x)
            loss.backward()
            optimizer.step()

        Args:
            x (torch.Tensor): Training data.
        
        Returns:
            (torch.Tensor, dict[str, any]): WNAE loss, training dictionary
            
            The training dictionary contains the following keys:

                - "loss" (torch.Tensor): the WNAE loss function
                - "reco_errors" (torch.Tensor): the reconstruction errors of the positive examples
                - "positive_energy" (float): the positive energy
                - "negative_energy" (float): the negative energy
                - "positive_z" (torch.Tensor): the latent representation of the positive examples
                - "negative_z" (torch.Tensor): the latent representation of the negative examples
                - "mcmc_data" (dict[str, any]): information about the MCMC, all without grad:

                    - "samples" (list[torch.Tensor]): MCMC samples, for each MCMC step
                    - "drift" (list[torch.Tensor]): drift terms, for each MCMC step
                    - "diffusion" (list[torch.Tensor]): diffusion terms, for each MCMC step
                    - "steps" (list[torch.Tensor]): MCMC steps, drift / temperature + diffusion, for each MCMC step

                    If running OMI, the following information is also available:

                    - "samples_z" (list[torch.Tensor]): same as `samples` but for the latent MCMC
                    - "drift_z" (list[torch.Tensor]): same as `drift` but for the latent MCMC
                    - "diffusion_z" (list[torch.Tensor]): same as `diffusion` but for the latent MCMC
                    - "steps_z" (list[torch.Tensor]): same as `steps` but for the latent MCMC
        """

        loss, _, _, training_dict = self.__wnae_step(x, mcmc_replay=True)
        return loss, training_dict

    def train_step_ae(self, x, run_mcmc=False):
        """Standard AE training step.
        
        Returns the AE loss and information about the training on the provided data.
        Use as:
        
        .. code-block:: python

            model.train()
            optimizer.zero_grad()
            loss, training_dict = model.train_step_ae(x)
            loss.backward()
            optimizer.step()

        Args:
            x (torch.Tensor): Training data.
            run_mcmc (bool, optional, default=True): Whether or not to run the MCMC.
                The MCMC samples will not be used for the AE loss computation,
                but can be used for training diagnostic purposes.
        
        Returns:
            (torch.Tensor, dict[str, any]): AE loss, training dictionary
            
            The training dictionary contains the same information as
            described in :meth:`~wnae.WNAE.train_step`.
            If `run_mcmc` is False, then the dictionary does not contain the
            following keys:

                - "negative_energy"
                - "negative_z"
                - "mcmc_data"
        """

        _, loss, _, training_dict = self.__wnae_step(
            x,
            mcmc_replay=True,
            compute_emd=False,
            detach_negative_samples=True,
            run_mcmc=run_mcmc,
        )
        training_dict["loss"] = loss.item()  # overwrite WNAE loss by AE loss
        return loss, training_dict

    def train_step_nae(self, x):
        """Standard NAE training step.
        
        Returns the NAE loss and information about the training on the provided data.
        Use as:
        
        .. code-block:: python

            model.train()
            optimizer.zero_grad()
            loss, training_dict = model.train_step_nae(x)
            loss.backward()
            optimizer.step()

        Args:
            x (torch.Tensor): Training data.
        
        Returns:
            (torch.Tensor, dict[str, any]): NAE loss, training dictionary
            
            The training dictionary contains the same information as
            described in :meth:`~wnae.WNAE.train_step`.
        """

        _, _, loss, training_dict = self.__wnae_step(
            x,
            mcmc_replay=True,
            compute_emd=False,
            detach_negative_samples=True,
        )
        training_dict["loss"] = loss.item()  # overwrite WNAE loss by NAE loss
        return loss, training_dict

    def validation_step(self, x):
        """Perform validation step.
        
        Performs validation step and return information about the validation
        on the provided data.

        Use as:
        
        .. code-block:: python

            model.eval()
            validation_dict = model.validation_step(x)

        Args:
            x (torch.Tensor): Validation data.
        
        Returns:
            dict[str, any]: Returns a dictionary with the same information 
            as described in :meth:`~wnae.WNAE.train_step`.
        """

        return self.__wnae_step(x, mcmc_replay=False, compute_emd=True)[3]

    def evaluate(self, x):
        """Run bare evaluation of the model without running the MCMC.
        
        Use as:
        
        .. code-block:: python

            model.eval()
            validation_dict = model.validation_step(x)

        Args:
            x (torch.Tensor): Validation data.
        
        Returns:
            dict[str, any]
        """

        return self.__wnae_step(x, mcmc_replay=False, compute_emd=False)[3]

    def run_mcmc(self, x=None, replay=False, all_steps=False):
        """Run MCMC and return MCMC samples.

        Args:
            x (torch.Tensor or None, optional, default=None): Initial
                starting points for the MCMC. If None, the MCMC will be
                initialized from the replay buffer for PCD and randomly for
                OMI. Must be not None for CD. 
            replay (bool, optional, default=False): Whether or not to add
                the final state of th MCMC to the replay buffer when the
                MCMC algorithm is PCD.
            all_steps (bool, optional, default=False): Set to True to return
                the samples for all the MCMC steps, or False to only get
                the final MCMC samples.
        
        Returns:
            torch.Tensor: The MCMC samples, without grad.
            If all_steps, the tensor has three index, else two index.
            The last index is the step number. The first two are the example
            number and the MCMC coordinates in the input feature space.
        """

        if self.sampling == "cd" and x is None:
            raise ValueError

        if x is None:
            mcmc_data = self.__sample(replay=replay)
        else:
            mcmc_data = self.__sample_x(x0=x, replay=replay)
        
        if all_steps:
            return torch.dstack(mcmc_data["samples"])
        else:
            return mcmc_data["sample_x"].detach().cpu()

