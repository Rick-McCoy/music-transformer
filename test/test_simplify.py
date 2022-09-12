import unittest

import torch

from config.config import (
    DRUM_LIMIT,
    NOTE_LIMIT,
    NUM_TOKEN,
    PROGRAM_LIMIT,
    SPECIAL_LIMIT,
    TICK_LIMIT,
)
from model.simplify import SimplifyClass, SimplifyScore


class TestSimplifyClass(unittest.TestCase):
    def test_simplify_class(self):
        simplify_class = SimplifyClass()
        random_indices = torch.randint(0, NUM_TOKEN, (1000,))
        simplified_indices = simplify_class(random_indices)
        self.assertEqual(simplified_indices.shape, (1000,))
        random_indices_list = random_indices.tolist()
        simplified_indices_list = simplified_indices.tolist()
        for random_index, simplified_index in zip(random_indices_list, simplified_indices_list):
            if random_index < SPECIAL_LIMIT:
                self.assertEqual(simplified_index, 0)
            elif random_index < PROGRAM_LIMIT:
                self.assertEqual(simplified_index, 1)
            elif random_index < DRUM_LIMIT:
                self.assertEqual(simplified_index, 2)
            elif random_index < NOTE_LIMIT:
                self.assertEqual(simplified_index, 3)
            elif random_index < TICK_LIMIT:
                self.assertEqual(simplified_index, 4)
            else:
                raise ValueError("Invalid index")


class TestSimplifyScore(unittest.TestCase):
    def test_simplify_score(self):
        simplify_score = SimplifyScore()
        random_score = torch.rand((1000, NUM_TOKEN))
        random_normalized_score = random_score / random_score.sum(dim=1, keepdim=True)
        simplified_score = simplify_score(random_normalized_score)
        self.assertEqual(simplified_score.shape, (1000, 5))
        for random_row, simplified_row in zip(random_normalized_score, simplified_score):
            self.assertAlmostEqual(
                simplified_row[0].item(), random_row[:SPECIAL_LIMIT].sum().item()
            )
            self.assertAlmostEqual(
                simplified_row[1].item(), random_row[SPECIAL_LIMIT:PROGRAM_LIMIT].sum().item()
            )
            self.assertAlmostEqual(
                simplified_row[2].item(), random_row[PROGRAM_LIMIT:DRUM_LIMIT].sum().item()
            )
            self.assertAlmostEqual(
                simplified_row[3].item(), random_row[DRUM_LIMIT:NOTE_LIMIT].sum().item()
            )
            self.assertAlmostEqual(
                simplified_row[4].item(), random_row[NOTE_LIMIT:TICK_LIMIT].sum().item()
            )
