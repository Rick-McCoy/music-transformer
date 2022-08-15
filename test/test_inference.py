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
        checkpoint_1 = temp_dir / "epoch=0-val_loss=0.1.ckpt"
        checkpoint_1.touch()
        checkpoint_2 = temp_dir / "epoch=1-val_loss=0.2.ckpt"
        checkpoint_2.touch()
        checkpoint_3 = temp_dir / "epoch=2-val_loss=0.3.ckpt"
        checkpoint_3.touch()
        checkpoint_4 = temp_dir / "epoch=3-val_loss=0.4.ckpt"
        checkpoint_4.touch()
        checkpoint_5 = temp_dir / "epoch=4-val_loss=0.5.ckpt"
        checkpoint_5.touch()
        best_checkpoint = find_best_checkpoint(temp_dir)
        self.assertEqual(best_checkpoint, checkpoint_1)
        checkpoint_1.unlink()
        checkpoint_2.unlink()
        checkpoint_3.unlink()
        checkpoint_4.unlink()
        checkpoint_5.unlink()
        temp_dir.rmdir()
