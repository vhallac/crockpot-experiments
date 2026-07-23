import math
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

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


class FakeConfig:
    def __init__(self, **attrs):
        for key, value in attrs.items():
            setattr(self, key, value)


class FakeModel:
    def __init__(self, config):
        self.config = config


class FakeLM:
    def __init__(self, config):
        self.model = FakeModel(config)


class PositionContentGateTests(unittest.TestCase):
    def test_g2_gpt2_layer0_passes_and_shuffled_correspondence_fails(self):
        lm = FakeLM(FakeConfig(model_type='gpt2', n_positions=1024, n_ctx=1024))
        self.assertTrue(pc._is_architectural_one_case(lm, layer=0, variant='pre'))
        y = np.arange(128, dtype=float)
        x = np.column_stack([y, y * 0.5, np.sin(y / 7.0), np.cos(y / 11.0)])
        r2 = pc._ridge_cv_r2(x, y, folds=5, seed=0, alphas=np.logspace(-2, 4, 13))
        perturbed_r2 = pc._g2_perturbed_ridge_r2(x, y, seed=0)
        self.assertGreaterEqual(r2, 0.9)
        self.assertFalse(perturbed_r2 >= 0.9)

    def test_g2_not_applicable_for_nope_reports_not_applicable(self):
        lm = FakeLM(FakeConfig(model_type='nope-gpt'))
        self.assertFalse(pc._is_architectural_one_case(lm, layer=0, variant='pre'))
        self.assertEqual(pc._gate_status(pd.DataFrame(), 'G2_architectural_one'), 'NOT_APPLICABLE')

    def test_all_architectural_gates_inapplicable_raises(self):
        with self.assertRaisesRegex(RuntimeError, 'no applicable architectural gates'):
            pc._raise_if_no_architectural_gate_applied({
                'gate_g1_pass': 'NOT_APPLICABLE',
                'gate_g2_pass': 'NOT_APPLICABLE',
            })

    def test_gates_evaluated_counts_match_gates_csv(self):
        gates = pd.DataFrame([
            {'gate': 'G2_architectural_one', 'pass': True, 'perturbation_can_fail': True},
            {'gate': 'G2_architectural_one', 'pass': True, 'perturbation_can_fail': True},
            {'gate': 'G1_architectural_zero', 'pass': True, 'perturbation_can_fail': True},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = f'{tmp}/gates.csv'
            gates.to_csv(path, index=False)
            reread = pd.read_csv(path)
        self.assertEqual(pc._gates_evaluated(reread), {
            'G1_architectural_zero': 1,
            'G2_architectural_one': 2,
        })


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



class M16BuildTests(unittest.TestCase):
    def test_m16_builds_probed_only_marker_stimuli_with_interior_target(self):
        from kaddress.scripts import m16_discriminator as m16

        tokenizer = FakeTokenizer()
        build = m16.build_stimuli(tokenizer, repetitions=128, limit_stimuli=1)
        self.assertEqual(len(build), 1)
        stim = build[0]
        self.assertEqual(len(stim.markers), 4)
        self.assertEqual(len(stim.marker_positions), 4)
        self.assertEqual(len(stim.continuation_positions), 128)
        self.assertGreater(stim.target_rep, 0)
        self.assertLess(stim.target_rep, 127)
        self.assertNotEqual(stim.target_rep, stim.donor_rep)
        self.assertNotEqual(stim.altered_rep, stim.target_rep)
        self.assertEqual(stim.readout_pos, len(stim.input_ids) - 1)
        marker_position_set = set(stim.marker_positions)
        self.assertEqual(len(marker_position_set), 4)
        marked_reps = {i for i, pos in enumerate(stim.continuation_positions) if pos in marker_position_set}
        self.assertEqual(marked_reps, {stim.target_rep, stim.donor_rep, stim.altered_rep, stim.readout_rep})

    def test_m16_induction_metrics_use_marker_positions_as_match_plus_one(self):
        from kaddress.scripts import m16_discriminator as m16
        import torch

        tokenizer = FakeTokenizer()
        stim = m16.build_stimuli(tokenizer, repetitions=8, limit_stimuli=1)[0]
        attn = torch.zeros(len(stim.input_ids))
        for pos in stim.continuation_positions:
            attn[pos] = 0.1
        metrics = m16._induction_metrics(stim, attn)
        self.assertAlmostEqual(metrics['induction_match_plus_one_mass'], 0.8, places=6)
        self.assertAlmostEqual(metrics['induction_most_recent_match_plus_one_mass'], 0.1, places=6)

    def test_m16_classification_requires_attention_and_output_above_noise_for_addressing(self):
        from kaddress.scripts import m16_discriminator as m16

        rows = pd.DataFrame([
            {'layer': 0, 'head': 0, 'patch_mode': 'baseline', 'target_attention_delta': 0.0, 'donor_prob_delta': 0.0, 'induction_match_plus_one_mass': 0.01, 'transitivity_altered_marker_prob': 0.0, 'transitivity_altered_marker_rank': 100},
            {'layer': 0, 'head': 0, 'patch_mode': 'k', 'target_attention_delta': 0.20, 'donor_prob_delta': 0.0, 'induction_match_plus_one_mass': 0.01, 'transitivity_altered_marker_prob': 0.0, 'transitivity_altered_marker_rank': 100},
            {'layer': 0, 'head': 0, 'patch_mode': 'v', 'target_attention_delta': 0.0, 'donor_prob_delta': 0.0, 'induction_match_plus_one_mass': 0.01, 'transitivity_altered_marker_prob': 0.0, 'transitivity_altered_marker_rank': 100},
            {'layer': 0, 'head': 0, 'patch_mode': 'both', 'target_attention_delta': 0.20, 'donor_prob_delta': 0.0, 'induction_match_plus_one_mass': 0.01, 'transitivity_altered_marker_prob': 0.0, 'transitivity_altered_marker_rank': 100},
            {'layer': 0, 'head': 0, 'patch_mode': 'noise', 'target_attention_delta': 0.19, 'donor_prob_delta': 0.0, 'induction_match_plus_one_mass': 0.01, 'transitivity_altered_marker_prob': 0.0, 'transitivity_altered_marker_rank': 100},
        ])
        classified = m16._classification(rows, attention_margin=0.02, output_margin=1e-4)
        self.assertNotEqual(classified.iloc[0]['classification'], 'addressing')
        self.assertFalse(bool(classified.iloc[0]['g7_noise_controlled_attention_pass']))

    def test_m16_classification_requires_positive_k_attention_redirection(self):
        from kaddress.scripts import m16_discriminator as m16

        rows = pd.DataFrame([
            {'layer': 0, 'head': 0, 'patch_mode': 'baseline', 'target_attention_delta': 0.0, 'donor_prob_delta': 0.0, 'induction_match_plus_one_mass': 0.01, 'transitivity_altered_marker_prob': 0.0, 'transitivity_altered_marker_rank': 100},
            {'layer': 0, 'head': 0, 'patch_mode': 'k', 'target_attention_delta': -0.05, 'donor_prob_delta': 0.0, 'induction_match_plus_one_mass': 0.01, 'transitivity_altered_marker_prob': 0.0, 'transitivity_altered_marker_rank': 100},
            {'layer': 0, 'head': 0, 'patch_mode': 'v', 'target_attention_delta': 0.0, 'donor_prob_delta': 0.0, 'induction_match_plus_one_mass': 0.01, 'transitivity_altered_marker_prob': 0.0, 'transitivity_altered_marker_rank': 100},
            {'layer': 0, 'head': 0, 'patch_mode': 'both', 'target_attention_delta': -0.05, 'donor_prob_delta': 0.001, 'induction_match_plus_one_mass': 0.01, 'transitivity_altered_marker_prob': 0.0, 'transitivity_altered_marker_rank': 100},
            {'layer': 0, 'head': 0, 'patch_mode': 'noise', 'target_attention_delta': -0.19, 'donor_prob_delta': 0.0, 'induction_match_plus_one_mass': 0.01, 'transitivity_altered_marker_prob': 0.0, 'transitivity_altered_marker_rank': 100},
        ])
        classified = m16._classification(rows, attention_margin=0.02, output_margin=1e-4)
        self.assertNotEqual(classified.iloc[0]['classification'], 'addressing')
        self.assertFalse(bool(classified.iloc[0]['g7_noise_controlled_attention_pass']))


if __name__ == '__main__':
    unittest.main()
