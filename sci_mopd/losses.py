"""Causal scoring and the immediate, stop-gradient reverse-KL proxy (not sequence-unbiased)."""
from __future__ import annotations
import torch
SAMPLING={'do_sample':True,'temperature':1.0,'top_p':1.0,'top_k':0,'repetition_penalty':1.0,'no_repeat_ngram_size':0,'num_beams':1}
def freeze(model):
 model.eval()
 for p in model.parameters():p.requires_grad_(False)
 return model
def assistant_mask(input_ids,prompt_lengths,lengths):
 # Columns correspond to targets input_ids[:,1:], not unshifted input positions.
 positions=torch.arange(input_ids.shape[1]-1,device=input_ids.device)[None]+1
 return (positions>=prompt_lengths[:,None]) & (positions<lengths[:,None])
def token_logprobs(logits,input_ids):
 if logits.shape[:2]!=input_ids.shape:raise ValueError('full causal logits/input shape required')
 return logits[:,:-1].float().log_softmax(-1).gather(-1,input_ids[:,1:,None]).squeeze(-1)
def token_proxy(log_s_current,log_s_old,log_teacher,adv_clip=10.0):
 advantage=(log_teacher.detach()-log_s_old.detach())
 if adv_clip is not None:
  if adv_clip<=0:raise ValueError('adv_clip must be positive')
  advantage=advantage.clamp(-adv_clip,adv_clip)
 return -advantage.detach()*log_s_current

def domain_mean(losses,mask):
 if losses.shape!=mask.shape:raise ValueError('loss/mask shapes differ')
 count=mask.sum()
 if int(count)==0:raise ValueError('domain has no valid assistant tokens; do not silently drop it')
 return (losses*mask).sum()/count

def weighted_proxy(domain_scores,g,adv_clip=10.0):
 if set(domain_scores)!=set(g):raise ValueError('every weighted domain must be present')
 if any(v<0 for v in g.values()) or abs(sum(g.values())-1)>1e-7:raise ValueError('nonnegative fixed g must sum to one')
 total=0;metrics={}
 for d,pack in domain_scores.items():
  cur,old,teacher,mask=pack
  loss=domain_mean(token_proxy(cur,old,teacher,adv_clip),mask);total=total+g[d]*loss
  with torch.no_grad():metrics[d]={'valid_tokens':int(mask.sum()),'proxy':float(loss.detach()),'sampled_log_ratio_mean':float(domain_mean(old-teacher,mask)),
                               'fresh_score_max_difference':float((cur.detach()-old).abs()[mask].max())}
 return total,metrics
