import torch

import rootutils

import numpy as np

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.dataset.synthetic.normalized_dataset import NormalizedSyntheticDataset

class AutoregSyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 length: int, 
                 max_seq_len: int,
                 min_seq_len: int = 3,
                 dict_size: int = 255, 
                 beta: float = 30, 
                 seed: int = 777,
                 dtype: np.dtype = np.int32,) -> None:
        super().__init__()
        
        self.inner = NormalizedSyntheticDataset(
            length = length,
            max_seq_len = max_seq_len + 1,
            min_seq_len = min_seq_len + 1,
            dict_size = dict_size,
            beta = beta, 
            seed = seed,
            dtype = dtype,
        )

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int) -> dict:
        result = self.inner[index]
        sequence = result["sequence"]
        result["sequence"] = sequence[:-1]
        result["groundtruth"] = sequence[1:]
        result["length"] = result["length"] - 1 
        return result

import logging  
import unittest

class TestAutoregDataset(unittest.TestCase):
    def check_limitations(self, params: dict) -> None:
        logging.log(logging.DEBUG, params)
        dataset = AutoregSyntheticDataset(**params)

        for index in range(len(dataset)):
            sample = dataset[index]

            logging.log(logging.DEBUG, '\t' + str(sample))

            self.assertLessEqual(sample["length"], params["max_seq_len"])
            self.assertLessEqual(params["min_seq_len"], sample["length"])
            self.assertEqual(len(sample["sequence"]), params["max_seq_len"])
            self.assertEqual(len(sample["groundtruth"]), params["max_seq_len"])

            self.assertTrue(np.all(0 <= sample["sequence"]))
            self.assertTrue(np.all(sample["sequence"] <= params["dict_size"]))
            self.assertTrue(np.all(sample["sequence"][1:] == sample["groundtruth"][:-1]))

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
