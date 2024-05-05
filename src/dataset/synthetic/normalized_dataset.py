import torch

import rootutils

import numpy as np

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.dataset.synthetic.lazy_datatset import LazySyntheticDataset

class NormalizedSyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, length: int, 
                 max_seq_len: int,
                 min_seq_len: int = 1,
                 dict_size: int = 255, 
                 beta: float = 30, 
                 seed: int = 777,
                 dtype: np.dtype = np.int32) -> None:
        super().__init__()

        assert 1 < dict_size

        self.inner = LazySyntheticDataset(
            length = length, 
            max_seq_len = max_seq_len, 
            min_seq_len = min_seq_len, 
            dict_size = dict_size - 1, 
            beta = beta,
            seed = seed,
            dtype = dtype,
        )

        self.dict_size = dict_size
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int) -> dict:
        result = self.inner[index]
        pad_count = self.max_seq_len - result["length"]
        sequence = np.zeros(self.max_seq_len, dtype = np.int32)
        sequence[pad_count:] = result["sequence"] + 1
        result["sequence"] = sequence
        return result

import logging  
import unittest

class TestNormalizedDataset(unittest.TestCase):
    def check_limitations(self, params: dict) -> None:
        logging.log(logging.DEBUG, params)
        dataset = NormalizedSyntheticDataset(**params)

        for index in range(len(dataset)):
            sample = dataset[index]

            logging.log(logging.DEBUG, '\t' + str(sample))

            self.assertLessEqual(sample["length"], params["max_seq_len"])
            self.assertLessEqual(params["min_seq_len"], sample["length"])
            self.assertEqual(len(sample["sequence"]), params["max_seq_len"])

            self.assertTrue(np.all(0 <= sample["sequence"]))
            self.assertTrue(np.all(sample["sequence"] <= params["dict_size"]))

    def test_limitations(self) -> None:
        params = [
            {"length": 1, "min_seq_len": 1, "max_seq_len": 1, "dict_size": 2,},
            {"length": 7, "min_seq_len": 1, "max_seq_len": 1, "dict_size": 2,},
            {"length": 1, "min_seq_len": 1, "max_seq_len": 7, "dict_size": 2,},
            {"length": 1, "min_seq_len": 1, "max_seq_len": 1, "dict_size": 7,},
            {"length": 7, "min_seq_len": 1, "max_seq_len": 7, "dict_size": 7,},
            {"length": 7, "min_seq_len": 2, "max_seq_len": 7, "dict_size": 7,},
            {"length": 16, "min_seq_len": 2, "max_seq_len": 32, "dict_size": 64,},
        ]

        for p in params:
            self.check_limitations(p)


if __name__ == "__main__":
    unittest.main()
