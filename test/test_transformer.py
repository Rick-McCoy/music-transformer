import unittest

from hydra import initialize, compose
import torch

from model.transformer import Transformer


class TestTransformer(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="config")
            self.cfg = cfg
            self.transformer = Transformer(cfg)

    def test_transformer(self):
        tick = torch.zeros(8, self.cfg.model.data_len, dtype=torch.int64)
        pitch = torch.zeros(8, self.cfg.model.data_len, dtype=torch.int64)
        program = torch.zeros(8, self.cfg.model.data_len, dtype=torch.int64)
        velocity = torch.zeros(8, self.cfg.model.data_len, dtype=torch.int64)
        tick_out, pitch_out, program_out, velocity_out = self.transformer(
            tick, pitch, program, velocity)
        self.assertEqual(tick_out.size(),
                         (8, self.cfg.model.num_tick, self.cfg.model.data_len))
        self.assertEqual(
            pitch_out.size(),
            (8, self.cfg.model.num_pitch, self.cfg.model.data_len))
        self.assertEqual(
            program_out.size(),
            (8, self.cfg.model.num_program, self.cfg.model.data_len))
        self.assertEqual(
            velocity_out.size(),
            (8, self.cfg.model.num_velocity, self.cfg.model.data_len))
