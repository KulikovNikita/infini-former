import torch
import numpy as np

class LazySyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 length: int, 
                 max_seq_len: int,
                 min_seq_len: int = 1,
                 dict_size: int = 255, 
                 beta: float = 32, 
                 seed: int = 777,
                 dtype: np.dtype = np.int32) -> None:
        super().__init__()

        assert 0 < length
        assert 0.0 <= beta
        assert 0 < dict_size
        assert 0 <= min_seq_len
        assert min_seq_len <= max_seq_len
        assert dict_size <= np.iinfo(dtype).max
        
        self.dtype = dtype
        self.length, self.beta = length, beta
        self.dict_size, self.seed = dict_size, seed
        self.min_seq_len, self.max_seq_len = min_seq_len, max_seq_len
    
    def __len__(self) -> int:
        return self.length

    def _gen_by_seed(self, seed: int) -> np.random.Generator:
        bit_engine = np.random.MT19937(seed)
        return np.random.Generator(bit_engine)

    def _gen_length(self, gen: np.random.Generator) -> int:
        length = gen.exponential(self.beta, size = (1,))
        length = np.ceil(length).astype(dtype = self.dtype)
        return np.clip(length, self.min_seq_len, self.max_seq_len).item()

    def _gen_phase(self, gen: np.random.Generator) -> float:
        return gen.uniform(0.0, 2.0 * np.pi, size = (1,)).item()

    def _gen_frequency(self, gen: np.random.Generator) -> float:
        scale = self.min_seq_len * 2.0 * np.pi / self.max_seq_len
        return gen.exponential(scale, size = (1,)).item()

    def _gen_params(self, gen: np.random.Generator) -> dict:
        return {
            "phase": self._gen_phase(gen),
            "length": self._gen_length(gen),
            "frequency": self._gen_frequency(gen),
        }

    def _gen_seq(self, params: dict) -> np.array:
        xs = np.arange(start = 0, stop = params["length"])
        raw = np.sin(params["phase"] + xs * params["frequency"])
        raw = np.ceil(self.dict_size * 0.5 *  (raw + 1.0))
        raw = np.clip(raw, a_min = 0, a_max = self.dict_size)
        return raw.astype(dtype = self.dtype)

    def _get_by_seed(self, index: int, seed: int) -> dict:
        gen = self._gen_by_seed(seed)
        params = self._gen_params(gen)
        sequence = self._gen_seq(params)
        return {
            "sequence": sequence, 
            "index": index, 
            "seed": seed,
            **params,
        }

    def _get_seed(self, index: int) -> int:
        return abs(hash(str(index)) + self.seed)

    def __getitem__(self, index) -> dict:
        seed = self._get_seed(index)
        return self._get_by_seed(index, seed)

import logging  
import unittest

class TestLazyDataset(unittest.TestCase):
    def check_limitations(self, params: dict) -> None:
        logging.log(logging.DEBUG, params)
        dataset = LazySyntheticDataset(**params)

        for index in range(len(dataset)):
            sample = dataset[index]

            logging.log(logging.DEBUG, '\t' + str(sample))

            self.assertLessEqual(sample["length"], params["max_seq_len"])
            self.assertLessEqual(params["min_seq_len"], sample["length"])
            self.assertEqual(len(sample["sequence"]), sample["length"])

            self.assertTrue(np.all(0 <= sample["sequence"]))
            self.assertTrue(np.all(sample["sequence"] <= params["dict_size"]))

    def test_limitations(self) -> None:
        params = [
            {"length": 1, "min_seq_len": 1, "max_seq_len": 1, "dict_size": 1,},
            {"length": 7, "min_seq_len": 1, "max_seq_len": 1, "dict_size": 1,},
            {"length": 1, "min_seq_len": 1, "max_seq_len": 7, "dict_size": 1,},
            {"length": 1, "min_seq_len": 1, "max_seq_len": 1, "dict_size": 7,},
            {"length": 7, "min_seq_len": 1, "max_seq_len": 7, "dict_size": 7,},
            {"length": 7, "min_seq_len": 2, "max_seq_len": 7, "dict_size": 7,},
            {"length": 16, "min_seq_len": 2, "max_seq_len": 32, "dict_size": 64,},
        ]

        for p in params:
            self.check_limitations(p)


if __name__ == "__main__":
    unittest.main()
