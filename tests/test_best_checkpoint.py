import ast
import math
import os
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn

# Load only the checkpoint helper; the training entry point imports TensorBoard.
source = ast.parse(Path('script/run.py').read_text())
helper = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'save_best_checkpoint')
namespace = dict(torch=torch, nn=nn, math=math, os=os)
exec(compile(ast.Module(body=[helper], type_ignores=[]), 'script/run.py', 'exec'), namespace)
save_best_checkpoint = namespace['save_best_checkpoint']


class BestCheckpointTest(unittest.TestCase):
    def test_lower_equal_and_nan_f1_do_not_replace_best_weights(self):
        model = nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model_best.pth'
            best = -math.inf
            for epoch, f1 in enumerate([0.4, 0.8, 0.3, 0.8, math.nan]):
                with torch.no_grad():
                    model.weight.fill_(epoch)
                best = save_best_checkpoint(model, optimizer, None, epoch, (0.9, 0.7, f1), best, path)
            checkpoint = torch.load(path)
            self.assertEqual(best, 0.8)
            self.assertEqual(checkpoint['epoch'], 1)
            self.assertEqual(checkpoint['selection_split'], 'validation')
            self.assertEqual(checkpoint['model']['weight'].item(), 1)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            with torch.no_grad():
                model.weight.fill_(5)
            best = save_best_checkpoint(model, optimizer, None, 5, (1, 1, 0.9), best, path)
            self.assertEqual(torch.load(path)['epoch'], 5)
            self.assertEqual(best, 0.9)

if __name__ == '__main__':
    unittest.main()
