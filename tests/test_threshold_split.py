import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import torch
from torch_geometric.data import Data
from script import run_diff


def graph(pair=None):
    ei = torch.empty(2, 0, dtype=torch.long) if pair is None else torch.tensor(pair).reshape(2, 1)
    et = torch.zeros(ei.shape[1], dtype=torch.long)
    return Data(edge_index=torch.empty(2, 0, dtype=torch.long), edge_type=torch.empty(0, dtype=torch.long),
                target_edge_index=ei, target_edge_type=et, num_nodes=3, num_relations=2)


class ThresholdSplitTest(unittest.TestCase):
    def test_test_mode_uses_test_labels_and_reports_selection_source(self):
        prob = torch.full((3, 2, 3), .01)
        for h, t, value in [(0, 1, .8), (0, 2, .4), (1, 2, .6)]:
            prob[h, 0, t] = value
            prob[t, 1, h] = value
        vq = Mock()
        vq.vq_vae = SimpleNamespace(num_rel=2)
        vq.logits_from_head.return_value = torch.logit(prob)
        diff = Mock()
        diff.sample.return_value = torch.zeros(3, 2, 64)
        train, valid, test = graph(), graph([0, 1]), graph([0, 2])
        scores = {}
        for split in ('validation', 'test'):
            cfg = SimpleNamespace(train=SimpleNamespace(vq_chunk=3),
                                  evaluation={'assumption': 'cwa', 'threshold_split': split, 'threshold_steps': 30})
            with patch.object(run_diff, 'precompute_tokens_chunked', return_value=torch.zeros(3,2,64)):
                scores[split] = run_diff.evaluate(cfg, vq, diff, train, test, 'cpu', Mock(),
                                                test, valid, return_details=True)
            self.assertEqual(scores[split]['threshold_selection_split'], split)
        self.assertGreater(scores['validation']['threshold'], .6)
        self.assertLess(scores['test']['threshold'], .4)
        self.assertEqual(scores['validation']['test_f_tsp'], 0)
        self.assertGreater(scores['test']['test_f_tsp'], .6)
        self.assertIsNone(scores['test']['validation_f_tsp'])
        self.assertAlmostEqual(scores['test']['selection_f_tsp'], scores['test']['test_f_tsp'])

    def test_near_one_threshold_uses_float32_for_final_counts(self):
        prob = torch.full((3,2,3), .01)
        for h,t,value in [(0,1,.8),(0,2,1.0),(1,2,.99999976)]:
            prob[h,0,t]=value
            prob[t,1,h]=value
        vq=Mock();vq.vq_vae=SimpleNamespace(num_rel=2)
        vq.logits_from_head.return_value=torch.logit(prob)
        diff=Mock();diff.sample.return_value=torch.zeros(3,2,64)
        cfg=SimpleNamespace(train=SimpleNamespace(vq_chunk=3),
                            evaluation={'assumption':'cwa','threshold_split':'test','threshold_steps':30})
        with patch.object(run_diff,'precompute_tokens_chunked',return_value=torch.zeros(3,2,64)):
            result=run_diff.evaluate(cfg,vq,diff,graph(),graph([0,2]),'cpu',Mock(),
                                     graph([0,2]),graph([0,1]),return_details=True)
        self.assertGreater(result['threshold'], .99999)
        self.assertEqual(result['selection_f_tsp'], 1.0)
        self.assertEqual(result['test_f_tsp'], 1.0)

if __name__ == '__main__':
    unittest.main()
