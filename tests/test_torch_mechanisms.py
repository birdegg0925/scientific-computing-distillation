import torch
from sci_mopd.losses import assistant_mask,token_logprobs,token_proxy,domain_mean,weighted_proxy
from sci_mopd.smoke import run_smoke

def test_causal_shift_prompt_padding_and_real_eos_mask():
 ids=torch.tensor([[9,8,2,3,0],[4,1,0,0,0]]);prompts=torch.tensor([2,1]);lengths=torch.tensor([5,3]);mask=assistant_mask(ids,prompts,lengths)
 assert mask.tolist()==[[False,True,True,True],[True,True,False,False]]
 logits=torch.randn(2,5,10);lp=token_logprobs(logits,ids)
 assert torch.allclose(lp[0,1],logits[0,1].log_softmax(-1)[2])

def test_instant_proxy_gradient_direction_and_detached_teacher():
 z=torch.tensor([1.4,0.0],requires_grad=True);logp=z.log_softmax(0);old=logp.detach();q=torch.tensor([.5,.5],requires_grad=True);logq=q.log()
 loss=(old.exp()*token_proxy(logp,old,logq,adv_clip=None)).sum();loss.backward();g=z.grad.clone()
 z2=z.detach().clone().requires_grad_();p=z2.softmax(0);kl=(p*(z2.log_softmax(0)-torch.tensor([.5,.5]).log())).sum();kl.backward()
 assert torch.allclose(g,z2.grad,atol=1e-6);assert g[0]>0;assert q.grad is None

def test_fixed_domain_mean_not_global_token_reweighting():
 cur=torch.tensor([[-2.,-2.,-2.]],requires_grad=True);old=cur.detach();teacher=torch.tensor([[-1.,-1.,-1.]])
 pack={'analysis':(cur,old,teacher,torch.tensor([[True,True,True]])), 'code':(cur[:,:1],old[:,:1],torch.tensor([[1.]]),torch.tensor([[True]]))}
 loss,metrics=weighted_proxy(pack,{'analysis':.5,'code':.5},adv_clip=None)
 assert abs(float(loss.detach())-4.)<1e-6 # domain losses 2 and 6, equal g, not 3/4 vs 1/4.

def test_anchor_zero_at_start_and_empty_domain_rejected():
 import pytest
 x=torch.tensor([[-1.]],requires_grad=True);assert float(domain_mean(token_proxy(x,x.detach(),x.detach()),torch.tensor([[True]])).detach())==0
 with pytest.raises(ValueError):domain_mean(x,torch.tensor([[False]]))

def test_tiny_actual_rollout_update_freeze_and_save_load():
 assert run_smoke()['all_passed']
