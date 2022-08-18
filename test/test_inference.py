import random
import unittest
import uuid
from pathlib import Path

import torch

from inference import find_best_checkpoint, top_p_sampling


class TestInference(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_top_p_sampling(self):
        logits_1 = torch.arange(10, dtype=torch.float32) * 100
        output_1 = top_p_sampling(logits_1, prob=0.9)
        self.assertEqual(output_1, torch.LongTensor([9]))
        logits_2 = torch.arange(start=10, end=1, step=-1, dtype=torch.float32) * 100
        output_2 = top_p_sampling(logits_2, prob=0.9)
        self.assertEqual(output_2, torch.LongTensor([0]))
        logits_3 = torch.zeros(10, dtype=torch.float32)
        random_index = torch.randint(0, 10, (1,)).item()
        assert isinstance(random_index, int)
        logits_3[random_index] = 1000
        output_3 = top_p_sampling(logits_3, prob=0.9)
        self.assertEqual(output_3, torch.LongTensor([random_index]))

    def test_find_best_checkpoint(self):
        temp_dir_uuid = uuid.uuid4()
        temp_dir = Path(f"./{temp_dir_uuid}")
        temp_dir.mkdir()
        losses = [random.random() for _ in range(10)]
        for i, loss in enumerate(losses):
            Path(temp_dir / f"epoch={i}-val_loss={loss}.ckpt").touch()
        best_checkpoint = find_best_checkpoint(temp_dir)
        min_index = min(range(len(losses)), key=lambda i: losses[i])
        self.assertEqual(
            best_checkpoint,
            Path(temp_dir / f"epoch={min_index}-val_loss={losses[min_index]}.ckpt"),
        )
        for i in range(10):
            Path(temp_dir / f"epoch={i}-val_loss={losses[i]}.ckpt").unlink()
        temp_dir.rmdir()

    def test_find_best_checkpoint_no_checkpoints(self):
        temp_dir_uuid = uuid.uuid4()
        temp_dir = Path(f"./{temp_dir_uuid}")
        temp_dir.mkdir()
        best_checkpoint = find_best_checkpoint(temp_dir)
        self.assertEqual(best_checkpoint, Path(""))
        temp_dir.rmdir()
