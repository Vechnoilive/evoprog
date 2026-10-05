import copy
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lab2


class Lab2Tests(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(lab2.DEFAULT_CONFIG)
        self.graph = lab2.generate_graph(self.config)

    def test_dataset_connected_and_reproducible(self):
        self.assertEqual(self.graph, lab2.generate_graph(self.config))
        self.assertEqual(len(self.graph['edges']), 80)
        reached = {0}
        while True:
            previous = reached.copy()
            for a, b in self.graph['edges']:
                if a in reached or b in reached:
                    reached.update([a, b])
            if reached == previous:
                break
        self.assertEqual(len(reached), 40)

    def test_coverage_independent_set_reference(self):
        rng = random.Random(92)
        masks = lab2.neighborhoods(self.graph)
        for _ in range(200):
            mask = rng.getrandbits(40)
            expected = set(lab2.selected(mask, 40))
            for a, b in self.graph['edges']:
                if mask & (1 << a):
                    expected.add(b)
                if mask & (1 << b):
                    expected.add(a)
            self.assertEqual(lab2.coverage(mask, masks), len(expected))

    def test_operators_and_repair(self):
        rng = random.Random(77)
        for _ in range(1000):
            a, b = rng.getrandbits(40), rng.getrandbits(40)
            child = lab2.uniform_crossover(a, b, 40, rng)
            self.assertEqual(child & (a & b), a & b)
            self.assertEqual(child & ~(a | b), 0)
            fixed = lab2.repair(child, 40, 5, rng)
            self.assertTrue(lab2.feasible(fixed, 40, 5))
            self.assertEqual(fixed & child, fixed)
            mutated = lab2.swap_mutation(fixed, 40, 5, 1, rng)
            self.assertTrue(lab2.feasible(mutated, 40, 5))
            if fixed.bit_count() == 5:
                self.assertEqual(mutated.bit_count(), 5)
                self.assertEqual((mutated ^ fixed).bit_count(), 2)
        self.assertEqual(lab2.bit_mutation(13, 4, 0, rng), 13)
        self.assertEqual(lab2.bit_mutation(13, 4, 1, rng), 2)
        self.assertFalse(lab2.feasible(-1, 40, 5))
        self.assertFalse(lab2.feasible(1 << 40, 40, 5))

    def test_exact_on_small_known_graph(self):
        graph = {'vertices': 6, 'max_posts': 2, 'coverage_radius': 1,
                 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]}
        result = lab2.exact_reference(graph)
        self.assertEqual(result['coverage'], 6)
        self.assertEqual(result['evaluations'], 15)

    def test_budget_feasibility_monotonicity_and_seed(self):
        self.config['evaluation_budget'] = 237
        for method in self.config['methods']:
            a, history = lab2.run_search(self.graph, self.config, method, 123)
            b, repeated = lab2.run_search(self.graph, self.config, method, 123)
            self.assertEqual(a['evaluations'], 237)
            self.assertEqual(history, repeated)
            self.assertEqual(a['mask'], b['mask'])
            self.assertTrue(lab2.feasible(a['mask'], 40, 5))
            values = [h['best_coverage'] for h in history]
            self.assertEqual(values, sorted(values))
            if method != 'penalty_bit':
                self.assertEqual(a['feasible_fraction'], 1)

    def test_penalty_dominates_infeasible_candidates(self):
        masks = lab2.neighborhoods(self.graph)
        rng = random.Random(11)
        for _ in range(1000):
            mask = rng.getrandbits(40)
            if mask.bit_count() > 5:
                score = lab2.coverage(mask, masks) - 41 * (mask.bit_count() - 5)
                self.assertLess(score, 0)


if __name__ == '__main__':
    unittest.main()
