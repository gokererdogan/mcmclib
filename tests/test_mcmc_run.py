"""
mcmclib

Unit tests for MCMCRun and related classes.

Created on Dec 2, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import unittest

from mcmclib.mcmc_run import *

class MCMCRunTest(unittest.TestCase):
    def setUp(self):
        self.r = MCMCRun(info='test', iteration_count=2, best_sample_count=2)

    def tearDown(self):
        self.r = None

    def test_record_iteration(self):
        self.r.record_iteration(0, 1, 1.0, 1.0, 'move1')
        self.r.record_iteration(1, 0, 1.0, 1.0, 'move2')
        self.assertTrue(np.all(self.r.run_log.loc[0] == [0, 1, 1.0, 1.0, 'move1']))
        self.assertTrue(np.all(self.r.run_log.loc[1] == [1, 0, 1.0, 1.0, 'move2']))
        self.assertRaises(KeyError, self.r.record_iteration, 2, 1, 1, 1, 'test')

    def test_add_sample(self):
        self.r.add_sample('sample1', 1.0, 1, 'move1')
        self.r.add_sample('sample2', 1.0, 5, 'move2')
        self.assertListEqual(list(self.r.samples[0]), ['sample1', 1.0, 1, 'move1'])
        self.assertListEqual(list(self.r.samples[1]), ['sample2', 1.0, 5, 'move2'])

    def test_add_best_sample(self):
        self.r.add_best_sample('sample1', 1.0, 1, 'move1')
        self.r.add_best_sample('sample2', 2.0, 5, 'move2')
        self.assertListEqual(list(self.r.best_samples[0]), ['sample1', 1.0, 1, 'move1'])
        self.assertListEqual(list(self.r.best_samples[1]), ['sample2', 2.0, 5, 'move2'])
        # this sample should not be added
        self.r.add_best_sample('sample3', 0.0, 15, 'move2')
        self.assertNotIn('sample3', self.r.best_samples.samples)
        # this one should be added
        self.r.add_best_sample('sample4', 4.0, 33, 'move1')
        self.assertIn('sample4', self.r.best_samples.samples)
        self.assertIn('sample2', self.r.best_samples.samples)

    def test_acceptance_rate_by_move(self):
        self.r.record_iteration(0, 1, 1.0, 1.0, 'move1')
        self.r.record_iteration(1, 0, 1.0, 1.0, 'move2')
        t = self.r.acceptance_rate_by_move()
        self.assertTrue(np.all(t[t.MoveType == 'move1'].AcceptanceRate == 1.0))
        self.assertTrue(np.all(t[t.MoveType == 'move2'].AcceptanceRate == 0.0))
