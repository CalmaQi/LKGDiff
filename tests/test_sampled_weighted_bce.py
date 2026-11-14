import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F
from ultra.stage12 import VQGANFCGraph


class SampledWeightedBCETest(unittest.TestCase):
    def test_sampled_weights_and_gradients_match_reference(self):
        torch.manual_seed(7)
        model = VQGANFCGraph(
            dim=8, codebook_size=8, num_rel=2, num_nodes=7,
            tokens=2, z_dim=2, num_quantizers=2, ema=False,
            num_neg_samples=3, beta=0.0, mode='vq', pos_weight_max=3.0,
        ).train()
        batch = torch.tensor([0, 1, 2])
        edges = torch.tensor([[0, 1, 2, 0], [3, 3, 4, 5]])
        rels = torch.tensor([0, 0, 1, 1])
        embedding = torch.randn(3, 2, 8)
        for explicit_weight in (None, 2.5):
            with self.subTest(explicit_weight=explicit_weight):
                captured = {}
                original = model.compute_sparse_logits_and_labels
                def capture(*args, **kwargs):
                    result = original(*args, **kwargs)
                    captured['sample'] = result
                    return result
                with patch.object(model, 'compute_sparse_logits_and_labels', side_effect=capture):
                    loss, stats = model(batch, embedding, edges, rels, pos_weight=explicit_weight)
                logits, targets, tails = captured['sample']
                full_labels = model.build_labels(batch, edges, rels, 3, 'cpu')
                if explicit_weight is None:
                    positives = torch.bincount(tails[targets > 0.5], minlength=7).float()
                    negatives = torch.bincount(tails[targets < 0.5], minlength=7).float()
                    weights = (negatives.clamp_min(1) / positives.clamp_min(1)).clamp(max=3)
                else:
                    weights = torch.full((7,), explicit_weight)
                expected = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weights[tails])
                torch.testing.assert_close(loss, expected)
                actual_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
                ref_grad = torch.autograd.grad(expected, logits)[0]
                torch.testing.assert_close(actual_grad, ref_grad)
                self.assertTrue(torch.isfinite(actual_grad).all())
                unweighted = F.binary_cross_entropy_with_logits(logits.detach(), targets)
                self.assertGreater(loss.item(), unweighted.item())
                self.assertEqual(stats['posw_max'], weights.max().item())
                self.assertLessEqual(logits.numel(), 4 * (1 + 3))


    def test_sampled_weights_are_local_and_capped(self):
        model = VQGANFCGraph(dim=8, codebook_size=8, num_rel=2, num_nodes=7,
                             pos_weight_max=5000000)
        targets = torch.cat([torch.ones(2), torch.zeros(12000)])
        tails = torch.tensor([3, 4] + [3] * 12000)
        weights = model.compute_sampled_pos_weight(targets, tails, torch.float32)
        self.assertEqual(weights[3].item(), 10000)
        self.assertEqual(weights[4].item(), 1)
        self.assertFalse(weights.requires_grad)
        override = model.compute_sampled_pos_weight(targets, tails, torch.float32, 20000.)
        self.assertEqual(override.max().item(), 10000)

    def test_known_positives_never_become_negatives(self):
        model = VQGANFCGraph(dim=8, codebook_size=8, num_rel=2, num_nodes=3,
                             num_neg_samples=100)
        heads = torch.tensor([0])
        edges = torch.tensor([[0, 0], [0, 1]])
        rels = torch.tensor([0, 0])
        x = torch.randn(1, 2, 8)
        with patch.object(model.inversedecoder, 'forward_sparse',
                          wraps=model.inversedecoder.forward_sparse) as spy:
            _, labels, _ = model.compute_sparse_logits_and_labels(
                x, heads, edges, rels, 1, 'cpu', return_tail_indices=True)
        _, b, r, t = spy.call_args.args
        known = model.build_labels(heads, edges, rels, 1, 'cpu')
        self.assertTrue(torch.equal(labels, known[b, r, t]))
        self.assertTrue((t[labels == 0] == 2).all())

    def test_empty_graph_sampled_loss_is_finite(self):
        model = VQGANFCGraph(dim=8, codebook_size=8, num_rel=2, num_nodes=3,
                             tokens=2, z_dim=2, ema=False, num_neg_samples=10,
                             beta=0, mode='vq')
        loss, stats = model(torch.tensor([0]), torch.randn(1, 2, 8),
                            torch.empty(2, 0, dtype=torch.long), torch.empty(0, dtype=torch.long))
        self.assertEqual(stats['loss_recon'], 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_density_losses_backpropagate_with_negative_sampling(self):
        for flag, stat in [('use_sparsity_loss', 'loss_sparse'), ('use_degree_loss', 'loss_degree')]:
            with self.subTest(flag=flag):
                torch.manual_seed(7)
                model = VQGANFCGraph(
                    dim=8, codebook_size=8, num_rel=2, num_nodes=7,
                    tokens=2, z_dim=2, num_quantizers=2, ema=False,
                    num_neg_samples=3, beta=0.0, mode='vq',
                ).train()
                embedding = torch.randn(3, 2, 8)
                batch = torch.tensor([0, 1, 2])
                edges = torch.tensor([[0, 1, 2, 0], [3, 3, 4, 5]])
                rels = torch.tensor([0, 0, 1, 1])
                def run(enabled):
                    setattr(model, flag, enabled)
                    model.zero_grad(set_to_none=True)
                    torch.manual_seed(17)
                    loss, stats = model(batch, embedding, edges, rels)
                    loss.backward()
                    grad = model.inversedecoder.tail_bias.grad.clone()
                    return stats, grad
                off, grad_off = run(False)
                on, grad_on = run(True)
                self.assertEqual(off[stat], 0.0)
                self.assertGreater(on[stat], 0.0)
                self.assertTrue(torch.isfinite(grad_on).all())
                self.assertGreater((grad_on - grad_off).abs().max().item(), 1e-8)


if __name__ == '__main__':
    unittest.main()
