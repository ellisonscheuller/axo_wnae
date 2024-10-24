import random
import torch
from wnae._logger import log


class SampleBuffer:
    """Sample buffer for MCMC with persistent algorithms."""

    def __init__(self, max_samples=10000, replay_ratio=0.95):
        self.max_samples = max_samples
        self.buffer = []
        self.replay_ratio = replay_ratio

    def __len__(self):
        return len(self.buffer)

    def push(self, samples):
        samples = samples.detach().to('cpu')

        for sample in samples:
            self.buffer.append(sample)

            if len(self.buffer) > self.max_samples:
                self.buffer.pop(0)

    def get(self, n_samples):
        if n_samples > len(self.buffer):
            log.warning(f"Sampling {n_samples} from buffer with size {len(self.buffer)}!")
        samples = random.choices(self.buffer, k=n_samples)
        samples = torch.stack(samples, 0)
        return samples

