"""Same-origin specialist, mixed-gold, or sequential SFT construction. No admission is assumed."""
from __future__ import annotations
import argparse,random,time
from pathlib import Path
from .common import read_jsonl,write_json,file_sha
from .hf import BASE_ID,load_tokenizer,base_identity,load_model,sft_batch,save_student,library_versions

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--model',default=BASE_ID)
 p.add_argument('--stage',choices=['analysis','code','primary','anchor'],default='primary');p.add_argument('--pool',choices=['teacher_sft','distill','anchor'],default='teacher_sft');p.add_argument('--schedule',choices=['mixed','staged'],default='mixed')
 p.add_argument('--steps',type=int,default=20);p.add_argument('--max-length',type=int,default=2048);p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--seed',type=int,default=7);p.add_argument('--device',default='cpu');p.add_argument('--rank',type=int,default=4);a=p.parse_args(argv)
 if a.steps<1:raise ValueError('steps must be positive')
 import torch
 from .losses import token_logprobs,domain_mean
 torch.manual_seed(a.seed);random.seed(a.seed);torch.set_num_threads(2)
 rows=[r for r in read_jsonl(a.data) if r['split']=='train' and r['pool']==a.pool and (r['stage'] in ['analysis','code'] if a.stage=='primary' else r['stage']==a.stage)]
 if not rows:raise ValueError('SFT selection is empty')
 t=load_tokenizer(a.model);identity=base_identity(a.model,t);m=load_model(a.model,trainable=True,rank=a.rank,device=a.device);optimizer=torch.optim.AdamW([q for q in m.parameters() if q.requires_grad],lr=a.lr)
 rng=random.Random(a.seed);logs=[];stages={s:[r for r in rows if r['stage']==s] for s in sorted(set(r['stage'] for r in rows))};start=time.perf_counter()
 for step in range(a.steps):
  if a.schedule=='staged' and a.stage=='primary':stage='analysis' if step<a.steps//2 else 'code';row=rng.choice(stages[stage])
  else:row=rng.choice(rows)
  m.eval();ids,att,mask=sft_batch(t,[row],a.max_length,a.device);logp=token_logprobs(m(input_ids=ids,attention_mask=att,use_cache=False).logits,ids)
  loss=-domain_mean(logp,mask);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([q for q in m.parameters() if q.requires_grad],1.);optimizer.step()
  logs.append({'step':step+1,'problem_id':row['problem_id'],'stage':row['stage'],'loss':float(loss.detach()),'assistant_tokens':int(mask.sum())})
  print(f"sft step={step+1}/{a.steps} stage={row['stage']} loss={float(loss.detach()):.6f}",flush=True)
 provenance={}
 for row in rows:
  kind=row.get('target_provenance',{}).get('kind','synthetic_gold');provenance[kind]=provenance.get(kind,0)+1
 metadata={'kind':'same-origin SFT with explicitly recorded target provenance','target_provenance_counts':provenance,'input_data_sha256':file_sha(a.data),'stage':a.stage,'pool':a.pool,'schedule':a.schedule,'steps':a.steps,'dropout_disabled':True,'args':vars(a)}
 save_student(m,t,a.out,identity,metadata);write_json(Path(a.out)/'sft_run.json',{'metadata':metadata,'versions':library_versions(),'seconds':time.perf_counter()-start,'logs':logs,'available_rows':len(rows),'human_admitted_teacher':False})
if __name__=='__main__':main()
