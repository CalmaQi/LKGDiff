import ast
from pathlib import Path
import unittest
import torch
from ultra.diffnew1 import TokenDiffusionMR, mr_ou_schedule


class OUDiffusionTest(unittest.TestCase):
    def test_ou_schedule_and_clean_endpoint(self):
        a, s = mr_ou_schedule(7, kappa=4., sigma_inf=1.5)
        tau = torch.linspace(0,1,7)
        torch.testing.assert_close(a, torch.exp(-4*tau))
        torch.testing.assert_close(s, 1.5*torch.sqrt(1-torch.exp(-8*tau)))
        model = TokenDiffusionMR(4,3,T=7,hidden=8,layers=1,time_dim=8,
                                 use_ou=True,use_relation_attn=False,use_node_attn=False)
        x = torch.randn(1,4,3,4)
        corrupted, _ = model.q_sample(x,torch.tensor([0]),torch.randn_like(x))
        torch.testing.assert_close(corrupted,x,rtol=0,atol=0)

    def test_ou_matches_vqdiff_forward_loss_and_reverse_sampling(self):
        path = Path('../VQDiff/ultra/diffnew1.py')
        if not path.exists(): self.skipTest('VQDiff reference checkout is unavailable')
        tree = ast.parse(path.read_text()); body=[]
        for node in tree.body:
            if isinstance(node,ast.ClassDef) and node.name=='EMA': break
            if isinstance(node,ast.ImportFrom) and (node.level or node.module=='ultra.base_nbfnet'): continue
            body.append(node)
        ns={};exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),ns)
        kwargs=dict(z_dim=4,tokens=3,T=7,hidden=8,layers=1,time_dim=8,heads=1,dim_head=4,dropout=0)
        ref=ns['TokenDiffusionMR'](**kwargs)
        model=TokenDiffusionMR(**kwargs,use_ou=True,use_relation_attn=False,use_node_attn=False)
        model.load_state_dict(ref.state_dict(),strict=True)
        ref.eval();model.eval()
        x=torch.randn(1,4,3,4);known=torch.randn_like(x);t=torch.tensor([4])
        for obj in (ref,model):
            torch.manual_seed(31)
            q,_=obj.q_sample(x,t,known)
            if obj is ref: qref=q
            else: torch.testing.assert_close(q,qref,rtol=0,atol=0)
        for eta in (0., .9):
            ref.eta=model.eta=eta
            torch.manual_seed(32);expected=ref.sample(known,heads=torch.tensor([0,2,3]))
            torch.manual_seed(32);actual=model.sample(known,heads=torch.tensor([0,2,3]))
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.manual_seed(33);loss_ref=ref.train_step(x,t,known)[0]
        torch.manual_seed(33);loss=model.train_step(x,t,known)[0]
        torch.testing.assert_close(loss,loss_ref,rtol=0,atol=0)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_ou_off_preserves_previous_ddpm(self):
        path=Path('experiments/cfamily_stage2_ablation_10k_20261114/no_attn_ffn_ratio095_testthr_gpu7/code/ultra/diffnew1.py')
        if not path.exists(): self.skipTest('Pre-OU source snapshot unavailable')
        ns={};exec(compile(path.read_text(),str(path),'exec'),ns)
        kwargs=dict(z_dim=4,tokens=3,T=7,hidden=8,layers=1,time_dim=8,heads=1,dim_head=4,
                    use_relation_attn=False,use_node_attn=False)
        ref=ns['TokenDiffusionMR'](**kwargs).eval();model=TokenDiffusionMR(**kwargs,use_ou=False).eval()
        model.load_state_dict(ref.state_dict(),strict=True)
        x=torch.randn(1,4,3,4);known=torch.randn_like(x)
        torch.manual_seed(11);expected=ref.q_sample(x,torch.tensor([5]),known)[0]
        torch.manual_seed(11);actual=model.q_sample(x,torch.tensor([5]),known)[0]
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.manual_seed(12);expected=ref.sample(known)
        torch.manual_seed(12);actual=model.sample(known)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

if __name__=='__main__': unittest.main()
