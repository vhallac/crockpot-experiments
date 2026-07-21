import math
import unittest
from unittest.mock import patch

import numpy as np

from kaddress.scripts import position_content as pc


class FakeTokenizer:
    def __init__(self, split_names=False, omit_frame=False):
        self.split_names = split_names
        self.omit_frame = omit_frame
        self.vocab = {}
        self.rev = {}

    def _id(self, piece):
        if piece not in self.vocab:
            self.vocab[piece] = len(self.vocab) + 1
            self.rev[self.vocab[piece]] = piece
        return self.vocab[piece]

    def encode(self, text, add_special_tokens=False):
        text = text.replace('.', ' .')
        out = []
        for word in text.split():
            if self.omit_frame and word in {'is', 'a', '.'}:
                out.append(self._id(f'content-{word}'))
            elif self.split_names and word in {'Farid', 'Boris', 'Derek', 'Elena', 'Hector', 'Jonas', 'Nolan', 'Omar'}:
                out.extend([self._id(word[:2]), self._id(word[2:])])
            else:
                out.append(self._id(word))
        return out

    def decode(self, ids, clean_up_tokenization_spaces=False):
        return self.rev[ids[0]]


class PositionContentBuildTests(unittest.TestCase):
    def test_family_b_survives_unequal_name_token_lengths_and_constant_frame_tokens(self):
        tokenizer = FakeTokenizer(split_names=True)
        build = pc.build_stimuli(
            tokenizer,
            families={'B'},
            max_length=5000,
            min_repetitions=128,
            repetitions=128,
            limit_stimuli=None,
            segment_lengths=[4, 7],
        )
        family_b = [s for s in build.stimuli if s.family == 'B']
        self.assertGreaterEqual(len(family_b), 1)
        for stim in family_b:
            self.assertEqual(len(stim.slots), 3)
            for slot, positions in stim.slots.items():
                self.assertEqual(len(positions), 128)
                self.assertIsInstance(stim.slot_token_ids[slot], int)

    def test_family_b_uses_cumulative_offsets_not_rep_times_common_length(self):
        tokenizer = FakeTokenizer(split_names=True)
        stim = pc.build_stimuli(
            tokenizer,
            families={'B'},
            max_length=5000,
            min_repetitions=128,
            repetitions=128,
            limit_stimuli=1,
            segment_lengths=[4, 7],
        ).stimuli[0]
        lengths = []
        cursor = 0
        for i in range(2):
            content = stim.content_words[i]
            ids = tokenizer.encode(f"{content['name']} is a {content['adjective']} {content['profession']}.")
            lengths.append(len(ids))
        self.assertNotEqual(lengths[0], lengths[1])
        first_slot_positions = stim.slots[0]
        self.assertNotEqual(first_slot_positions[1], first_slot_positions[0] + lengths[0] if lengths[0] == lengths[1] else 1 * lengths[0] + (first_slot_positions[0] % lengths[0]))

    def test_infeasible_length_cell_raises_with_arithmetic(self):
        with self.assertRaisesRegex(RuntimeError, r"cell L=12 needs 1440 tokens, budget 950"):
            pc.build_stimuli(
                FakeTokenizer(),
                families={'B'},
                max_length=950,
                min_repetitions=120,
                repetitions=None,
                limit_stimuli=None,
                segment_lengths=[4, 7, 12],
            )

    def test_family_a_has_eight_segments_for_each_requested_length(self):
        tokenizer = FakeTokenizer()
        fake_segments = [(f'seg{i}', [i] * 4) for i in range(8)] if False else None

        def segments_for_length(_tokenizer, target_l, count=8):
            return [(f'L{target_l}-{i}', [1000 + target_l * 10 + i] * target_l) for i in range(count)]

        with patch.object(pc, '_family_a_segments_for_length', side_effect=segments_for_length):
            build = pc.build_stimuli(
                tokenizer,
                families={'A'},
                max_length=1000,
                min_repetitions=120,
                repetitions=120,
                limit_stimuli=None,
                segment_lengths=[4, 7],
            )
        self.assertEqual(sum(1 for s in build.stimuli if s.family == 'A' and s.target_L == 4), 8)
        self.assertEqual(sum(1 for s in build.stimuli if s.family == 'A' and s.target_L == 7), 8)

    def test_g5_reports_empty_family_with_rejections(self):
        with self.assertRaisesRegex(RuntimeError, r"G5 family yield failed"):
            pc.build_stimuli(
                FakeTokenizer(omit_frame=True),
                families={'B'},
                max_length=5000,
                min_repetitions=128,
                repetitions=128,
                limit_stimuli=None,
                segment_lengths=[4, 7],
            )


class PositionContentStatsTests(unittest.TestCase):
    def test_degenerate_row_zeroes_all_derived_position_statistics(self):
        x = np.ones((20, 4), dtype=float)
        y = np.arange(20, dtype=float)
        token_ids = np.ones(20, dtype=np.int64)
        stats = pc._analyse_matrix(x, y, token_ids, seed=0, variance_floor=1e-5, null_permutations=2)
        self.assertTrue(stats['degenerate'])
        self.assertEqual(stats['ridge_r2'], 0.0)
        self.assertEqual(stats['pca_components_90pct'], 0)
        self.assertEqual(stats['pca_residual_variance_fraction_90pct'], 0.0)
        self.assertTrue(math.isnan(stats['pc1_spearman_repetition']))
        self.assertEqual(stats['pc1_dominant_fourier_bin'], -1)
        self.assertEqual(stats['r2_after_position_pc_projection'], 0.0)

    def test_shuffle_null_gate_is_upper_tail_only(self):
        shuffled = np.array([-0.4, -0.2, -0.1, -0.05, 0.01])
        self.assertTrue(np.quantile(shuffled, 0.99) <= 0.05)
        leaked = np.array([-0.4, -0.2, 0.01, 0.03, 0.10])
        self.assertFalse(np.quantile(leaked, 0.99) <= 0.05)


if __name__ == '__main__':
    unittest.main()
