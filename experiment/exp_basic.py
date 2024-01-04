import os

import torch


class ExperimentBasic:
    def __init__(self, config) -> None:
        self.config = config
        self._acquire_device()
        self._build_model()
        self._load_data()

    def _acquire_device(self):
        if self.config.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.config.device_id) if not self.config.use_multi_gpu else self.config.device_ids
            self.device = torch.device(f"cuda:{self.config.device_id}")
            print(f"Using GPU: cuda:{self.config.device_id}...")
        else:
            self.device = torch.device("cpu")
            print("Using CPU...")

    def _build_model(self):
        raise NotImplementedError

    def _load_data(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def calculate_one_batch(self):
        raise NotImplementedError
