import unittest
import torch
from torch import nn
from ultra.diffnew1 import MergeDenoiser, TokenDiffusionMR, ChannelMLP, LinearAttention, AxisTokenMixMLP


class DiffusionVariantsTest(unittest.TestCase):
    def test_three_variants_shape_and_backward(self):
        for mixer, ffn in [('linear_attention', False), ('linear_attention', True), ('tokenmix_mlp', True)]:
            with self.subTest(mixer=mixer, ffn=ffn):
                model = TokenDiffusionMR(z_dim=4, tokens=3, hidden=8, layers=1,
                                         time_dim=8, heads=1, dim_head=4, T=5,
                                         use_ffn=ffn, mixer=mixer, node_mixing_size=5)
                x = torch.randn(1, 4, 3, 4, requires_grad=True)
                y = model.net(x, torch.tensor([.3]), torch.zeros_like(x))
                self.assertEqual(y.shape, x.shape)
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertIsInstance(model.net.blocks[0][1], ChannelMLP if ffn else nn.Identity)
                expected = LinearAttention if mixer == 'linear_attention' else AxisTokenMixMLP
                self.assertIsInstance(model.net.relation_attn[0], expected)
                self.assertIsInstance(model.net.node_attn[0], expected)
                self.assertTrue(any(p.grad is not None for p in model.net.node_attn.parameters()))
                y_full = model.net(torch.randn(1,5,3,4), torch.tensor([.3]), torch.zeros(1,5,3,4))
                self.assertEqual(y_full.shape, (1,5,3,4))

    def test_both_attention_branches_can_be_disabled(self):
        model = TokenDiffusionMR(z_dim=4, tokens=3, hidden=8, layers=2,
                                 time_dim=8, T=5, use_ffn=True,
                                 use_relation_attn=False, use_node_attn=False)
        self.assertEqual(len(model.net.relation_attn), 0)
        self.assertEqual(len(model.net.node_attn), 0)
        self.assertTrue(all(isinstance(block[1], ChannelMLP) for block in model.net.blocks))
        x = torch.randn(1,4,3,4,requires_grad=True)
        y = model.net(x, torch.tensor([.3]), torch.zeros_like(x))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_token_mixer_changes_other_positions(self):
        torch.manual_seed(1)
        for axis, length in [('tokens', 3), ('nodes', 5)]:
            mix = AxisTokenMixMLP(length, axis)
            x = torch.randn(1, 5, 3, 4)
            y = mix(x)
            x[:, 0, 0, :] += 2
            delta = (mix(x) - y).abs()
            self.assertGreater(delta[:,0,1,:].max().item() if axis == 'tokens' else delta[:,1,0,:].max().item(), 1e-6)

if __name__ == '__main__':
    unittest.main()
