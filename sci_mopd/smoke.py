"""Actual random-tiny-Torch mechanism checks; not a pretrained-model quality experiment."""
from __future__ import annotations
import argparse,copy,tempfile
from pathlib import Path
from .common import write_json

def run_smoke():
 import torch
 from torch import nn
 from types import SimpleNamespace
 from .hf import sample_fresh,pack_sequences
 from .losses import freeze,token_logprobs,weighted_proxy
 torch.manual_seed(71);torch.set_num_threads(1)
 class TinyLM(nn.Module):
  def __init__(self):
   super().__init__();self.emb=nn.Embedding(17,8);self.rnn=nn.GRU(8,8,batch_first=True);self.drop=nn.Dropout(.5);self.out=nn.Linear(8,17)
   with torch.no_grad():self.out.bias[0]=-100
  def forward(self,input_ids,attention_mask=None,use_cache=False):
   hidden,_=self.rnn(self.emb(input_ids));return SimpleNamespace(logits=self.out(self.drop(hidden)))
  def generate(self,input_ids,attention_mask,generation_config):
   self.last_config=generation_config;seq=input_ids
   for _ in range(generation_config.max_new_tokens):
    prob=self(seq).logits[:,-1].softmax(-1);n=torch.multinomial(prob,1);seq=torch.cat([seq,n],1)
    if int(n[0])==generation_config.eos_token_id:break
   return seq
 class TinyTokenizer:
  eos_token_id=0;pad_token_id=0;bos_token_id=3
  def apply_chat_template(self,messages,**kwargs):return [1,2,3]+[1+ord(c)%16 for c in messages[-1]['content'][:10]]
  def decode(self,ids,**kwargs):return ''.join(chr(65+i) for i in ids if i)
 student=TinyLM().eval();start=copy.deepcopy(student.state_dict());anchor=freeze(copy.deepcopy(student));math=freeze(copy.deepcopy(student));code=freeze(copy.deepcopy(student))
 with torch.no_grad():math.out.bias[1]+=.4;code.out.bias[2]+=.4
 teachers={'analysis':math,'code':code,'anchor':anchor};before={d:copy.deepcopy(m.state_dict()) for d,m in teachers.items()};tokenizer=TinyTokenizer();packs={};records={}
 for d in teachers:
  row={'prompt':d+' tiny synthetic smoke task'};r,ids,attention,mask=sample_fresh(student,tokenizer,[row],64,2)
  with torch.no_grad():old=token_logprobs(student(ids).logits,ids);target=token_logprobs(teachers[d](ids).logits,ids)
  current=token_logprobs(student(ids).logits,ids);packs[d]=(current,old,target,mask);records[d]=r[0]
 loss,metrics=weighted_proxy(packs,{'analysis':.4,'code':.4,'anchor':.2});loss.backward();grad=sum(float(p.grad.abs().sum()) for p in student.parameters() if p.grad is not None)
 optimizer=torch.optim.SGD(student.parameters(),lr=.05);optimizer.step()
 frozen=all(all(torch.equal(before[d][k],m.state_dict()[k]) for k in before[d]) and all(p.grad is None and not p.requires_grad for p in m.parameters()) for d,m in teachers.items())
 temp_root=Path(__file__).resolve().parents[1]/'verification';temp_root.mkdir(exist_ok=True)
 with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
  path=Path(tmp)/'student.pt';torch.save(student.state_dict(),path);loaded=TinyLM().eval();loaded.load_state_dict(torch.load(path,weights_only=True));save_equal=all(torch.equal(student.state_dict()[k],loaded.state_dict()[k]) for k in student.state_dict())
 cfg=student.last_config;checks={'student_eval_dropout_disabled':not student.training,'teachers_frozen_unchanged':frozen,'nonzero_student_gradient':grad>0,'student_changed':any(not torch.equal(start[k],student.state_dict()[k]) for k in start),'single_student_save_load_exact':save_equal,
 'fresh_scores_match':all(v['fresh_score_max_difference']<1e-6 for v in metrics.values()),'two_token_prefix_kept_without_fake_eos':all(r['truncated'] and len(r['completion_ids'])==2 and r['completion_ids'][-1]!=0 for r in records.values()),
 'pure_sampling_profile':cfg.do_sample and cfg.temperature==1 and cfg.top_p==1 and cfg.top_k==0 and cfg.repetition_penalty==1 and cfg.num_beams==1}
 return {'checks':checks,'all_passed':all(checks.values()),'loss':float(loss.detach()),'gradient_l1':grad,'domain_metrics':metrics,'scope':'Mechanism checks on a randomly initialized PyTorch model with a 17-token vocabulary.'}
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--out',default='runs/torch_smoke.json');a=p.parse_args(argv);result=run_smoke();write_json(a.out,result);print(result)
 if not result['all_passed']:raise SystemExit(1)
if __name__=='__main__':main()
