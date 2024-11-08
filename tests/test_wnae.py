import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_s_curve
from sklearn.model_selection import train_test_split

from wnae import WNAE


DEVICE = torch.device('cpu')

WNAE_PARAMETERS_CD = {
    "sampling": "cd",
    "n_steps": 10,
    "step_size": None,
    "noise": 0.2,
    "temperature": 0.05,
    "bounds": (-3, 3),
    "mh": False,
    "initial_distribution": "gaussian",
}

WNAE_PARAMETERS_PCD = {
    "sampling": "pcd",
    "n_steps": 10,
    "step_size": None,
    "noise": 0.2,
    "temperature": 0.05,
    "bounds": (-3, 3),
    "mh": False,
    "initial_distribution": "gaussian",
    "replay": True,
    "replay_ratio": 0.95,
    "buffer_size": 10000,
}

WNAE_PARAMETERS_OMI = {
    "sampling": "omi",
    "n_steps": 10,
    "step_size": None,
    "noise": 0.2,
    "temperature": 0.05,
    "bounds": (-3, 3),
    "mh": False,
    "z_n_steps": 10,
    "z_step_size": None,
    "z_noise": 0.2,
    "z_temperature": 0.05,
    "z_bounds": (-3, 3),
    "z_mh": False,
    "initial_distribution": "gaussian",
    "replay": False,
    "spherical": True,
}


def __load_data():
    samples, labels = make_s_curve(n_samples=1000, noise=0.1)
    training_data, validation_data = train_test_split(
        samples[:, [0, 2]],
        test_size=0.2,
        shuffle=True,
    )

    
    def get_loader(data):
        data = torch.tensor(data.astype(np.float32)).to(DEVICE)
        sampler = torch.utils.data.RandomSampler(
            data_source=data,
            num_samples=2**13,
            replacement=True,
        )
        loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(data),
            batch_size=512,
            sampler=sampler,
        )

        return loader

    training_loader = get_loader(training_data)
    validation_loader = get_loader(validation_data)

    return training_loader, validation_loader


def __get_wnae_model(parameters):
    class Encoder(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.layer1 = nn.Linear(input_size, 32)
            self.layer2 = nn.Linear(32, 32)

        def forward(self, x):
            x = self.layer1(x)
            x = nn.functional.relu(x)
            x = self.layer2(x)
            x = nn.functional.relu(x)
            return x

    class Decoder(nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.layer1 = nn.Linear(32, 32)
            self.layer2 = nn.Linear(32, output_size)

        def forward(self, x):
            x = self.layer1(x)
            x = nn.functional.relu(x)
            x = self.layer2(x)
            return x

    model = WNAE(
        encoder=Encoder(input_size=2),
        decoder=Decoder(output_size=2),
        **parameters,
    )

    model.to(DEVICE)

    return model


def __run_training(model, loss_function, training_loader, validation_loader):

    n_epochs = 3

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=3e-4,
    )

    for i_epoch in range(n_epochs):

        # Train step
        model.train()
        n_batches = 0
        training_loss = 0
        for batch in training_loader:
            n_batches += 1
            x = batch[0]

            optimizer.zero_grad()
            if loss_function == "wnae":
                loss, training_dict = model.train_step(x)
            elif loss_function == "nae":
                loss, training_dict = model.train_step_nae(x)
            elif loss_function == "ae_mcmc":
                loss, training_dict = model.train_step_ae(x, run_mcmc=True, mcmc_replay=True)
            elif loss_function == "ae":
                loss, training_dict = model.train_step_ae(x, run_mcmc=False)

            assert loss.requires_grad
            assert torch.all(~torch.isnan(loss), axis=0)
            assert torch.all(~torch.isinf(loss), axis=0)
            assert not isinstance(training_dict["loss"], torch.Tensor)

            loss.backward()
            optimizer.step()

            training_loss += training_dict["loss"]

        training_loss /= n_batches

        # Validation step
        model.eval()
        n_batches = 0
        validation_loss = 0
        for batch in validation_loader:
            n_batches += 1
            x = batch[0]

            if loss_function == "wnae":
                validation_dict = model.validation_step(x)
            elif loss_function == "nae":
                validation_dict = model.validation_step_nae(x)
            elif loss_function == "ae_mcmc":
                validation_dict = model.validation_step_ae(x, run_mcmc=True)
            elif loss_function == "ae":
                validation_dict = model.validation_step_ae(x, run_mcmc=False)

            assert not isinstance(validation_dict["loss"], torch.Tensor)
            validation_loss += validation_dict["loss"]

        validation_loss /= n_batches


def test_standalone_methods():

    training_loader, validation_loader = __load_data()
    wnae_model = __get_wnae_model(WNAE_PARAMETERS_PCD)

    # Check evaluation
    data = next(iter(training_loader))[0]
    reco_error = wnae_model.evaluate(data)["reco_errors"]

    assert len(reco_error) == len(data)

    # Check standalone mcmc
    n_points = 10
    x = torch.rand(n_points, 1)
    y = torch.rand(n_points, 1)
    initial_state = torch.hstack((x, y))
    mcmc_samples = wnae_model.run_mcmc(x=initial_state, all_steps=True)

    assert mcmc_samples.shape[0] == initial_state.shape[0]
    assert mcmc_samples.shape[1] == initial_state.shape[1]
    assert mcmc_samples.shape[2] == WNAE_PARAMETERS_PCD["n_steps"] + 1


def test_ae_pcd_training():
    training_loader, validation_loader = __load_data()
    wnae_model = __get_wnae_model(WNAE_PARAMETERS_PCD)
    __run_training(wnae_model, "ae", training_loader, validation_loader)


def test_nae_pcd_training():
    training_loader, validation_loader = __load_data()
    wnae_model = __get_wnae_model(WNAE_PARAMETERS_PCD)
    __run_training(wnae_model, "nae", training_loader, validation_loader)


def test_wnae_pcd_training():
    training_loader, validation_loader = __load_data()
    wnae_model = __get_wnae_model(WNAE_PARAMETERS_PCD)
    __run_training(wnae_model, "wnae", training_loader, validation_loader)


def test_wnae_cd_training():
    training_loader, validation_loader = __load_data()
    wnae_model = __get_wnae_model(WNAE_PARAMETERS_CD)
    __run_training(wnae_model, "wnae", training_loader, validation_loader)


def test_nae_omi():
    training_loader, validation_loader = __load_data()
    wnae_model = __get_wnae_model(WNAE_PARAMETERS_OMI)
    __run_training(wnae_model, "nae", training_loader, validation_loader)


def test_nae_omi():
    training_loader, validation_loader = __load_data()
    wnae_model = __get_wnae_model(WNAE_PARAMETERS_OMI)
    __run_training(wnae_model, "wnae", training_loader, validation_loader)
