"""Unit test for `model/attention.py`"""
import unittest

from hydra import initialize, compose
import torch

from model.attention import RotaryAttention, RotaryTransformerLayer, TransformerLayer


class TestAttention(unittest.TestCase):
    """Tester for `model/attention.py`."""
    def setUp(self) -> None:
        with initialize(config_path="../config"):
            cfg = compose(config_name="main")
            self.cfg = cfg
            self.rotary_attention = RotaryAttention(d_model=cfg.model.d_model,
                                                    nhead=cfg.model.nhead,
                                                    dropout=cfg.model.dropout)
            self.rotary_transformer_layer = RotaryTransformerLayer(
                d_model=cfg.model.d_model,
                nhead=cfg.model.nhead,
                ff=cfg.model.ff,
                dropout=cfg.model.dropout)
            self.transformer_layer = TransformerLayer(
                d_model=cfg.model.d_model,
                nhead=cfg.model.nhead,
                ff=cfg.model.ff,
                dropout=cfg.model.dropout)

    def test_rotary_attention(self):
        """Tester for RotaryAttention.

        Simply runs a sample input through the attention.
        Checks for output shape."""
        data = torch.randn(8, self.cfg.model.data_len, self.cfg.model.d_model)
        temporal = torch.randn(8, self.cfg.model.data_len)
        mask = torch.zeros(self.cfg.model.data_len, self.cfg.model.data_len)
        output = self.rotary_attention(data, temporal, mask)
        self.assertEqual(output.size(),
                         (8, self.cfg.model.data_len, self.cfg.model.d_model))

    def test_rotary_transformer_layer(self):
        """Tester for RotaryTransformerLayer.

        Simply runs a sample input through the layer.
        Checks for output shape."""
        data = torch.randn(8, self.cfg.model.data_len, self.cfg.model.d_model)
        temporal = torch.randn(8, self.cfg.model.data_len)
        mask = torch.zeros(self.cfg.model.data_len, self.cfg.model.data_len)
        output, new_temporal, new_mask = self.rotary_transformer_layer(
            (data, temporal, mask))
        self.assertEqual(output.size(),
                         (8, self.cfg.model.data_len, self.cfg.model.d_model))
        self.assertTrue(torch.all(new_temporal == temporal))
        self.assertTrue(torch.all(new_mask == mask))

    def test_transformer_layer(self):
        """Tester for TransformerLayer.

        Simply runs a sample input through the layer.
        Checks for output shape."""
        data = torch.randn(8, self.cfg.model.data_len, self.cfg.model.d_model)
        mask = torch.zeros(self.cfg.model.data_len, self.cfg.model.data_len)
        output, new_mask = self.transformer_layer((data, mask))
        self.assertEqual(output.size(),
                         (8, self.cfg.model.data_len, self.cfg.model.d_model))
        self.assertTrue(torch.all(new_mask == mask))
