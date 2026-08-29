from __future__ import annotations

import unittest

import torch
from torch.utils.data import TensorDataset

from marble.core.base_datamodule import BaseDataModule


class SingleProcessDataLoaderTest(unittest.TestCase):
    @staticmethod
    def _module(num_workers: int) -> BaseDataModule:
        module = BaseDataModule(
            batch_size=2,
            num_workers=num_workers,
            train={},
            val={},
            test={},
        )
        dataset = TensorDataset(torch.arange(4))
        module.train_dataset = dataset
        module.val_dataset = dataset
        module.test_dataset = dataset
        return module

    def test_single_process_loaders_omit_prefetch_factor(self):
        module = self._module(num_workers=0)
        for loader in (
            module.train_dataloader(),
            module.val_dataloader(),
            module.test_dataloader(),
        ):
            self.assertEqual(loader.num_workers, 0)
            self.assertIsNone(loader.prefetch_factor)
            self.assertEqual(len(next(iter(loader))[0]), 2)

    def test_worker_loader_keeps_prefetch_factor(self):
        loader = self._module(num_workers=1).train_dataloader()
        self.assertEqual(loader.num_workers, 1)
        self.assertEqual(loader.prefetch_factor, 2)
