"""Single-checkpoint inference; oracle replays references, accept_all/keyword are model-free shortcut baselines."""
from __future__ import annotations
import argparse,time
from pathlib import Path
from .common import read_jsonl,write_jsonl,write_json,file_sha,object_sha
from .hf import BASE_ID,load_tokenizer,load_model,base_identity,prompt_ids,sample_fresh
from .shortcuts import BASELINES

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--engine',choices=['hf','oracle','accept_all','keyword'],default='hf');p.add_argument('--model',default=BASE_ID);p.add_argument('--adapter')
 p.add_argument('--decode',choices=['greedy','sample'],default='greedy');p.add_argument('--max-new-tokens',type=int,default=256);p.add_argument('--max-prompt-tokens',type=int,default=2048);p.add_argument('--seed',type=int,default=7);p.add_argument('--device',default='cpu');p.add_argument('--limit',type=int,default=0);a=p.parse_args(argv)
 rows=read_jsonl(a.data);rows=rows[:a.limit] if a.limit else rows;results=[];m=t=None;identity=None;adapter_sha=None;checkpoint_id=None
 destination=Path(a.out)
 if destination.suffix!='.jsonl':raise ValueError('--out must be a JSONL file path')
 if a.engine=='hf':
  import torch
  from transformers import GenerationConfig
  torch.manual_seed(a.seed);torch.set_num_threads(2);t=load_tokenizer(a.model);identity=base_identity(a.model,t);m=load_model(a.model,a.adapter,device=a.device,identity=identity)
  adapter_sha=file_sha(Path(a.adapter)/'adapter_model.safetensors') if a.adapter else None
  checkpoint_id=object_sha({'base_identity':identity,'adapter_sha256':adapter_sha,'adapter_config_sha256':file_sha(Path(a.adapter)/'adapter_config.json') if a.adapter else None})
 for i,row in enumerate(rows):
  start=time.perf_counter();count=None;truncated=False;termination='reference'
  if a.engine!='hf':
   # Neither branch is a language model: oracle returns the reference, the shortcut baselines read only the prompt.
   completion=row['target'] if a.engine=='oracle' else BASELINES[a.engine](row)
  elif a.decode=='sample':
   records,_,_,_=sample_fresh(m,t,[row],a.max_prompt_tokens,a.max_new_tokens,a.device);r=records[0];completion=r['completion'];count=r['generated_tokens'];truncated=r['truncated'];termination=r['termination']
  else:
   ids=prompt_ids(t,row)
   if len(ids)>a.max_prompt_tokens:raise ValueError('prompt exceeds limit; do not silently truncate')
   import torch
   from .losses import SAMPLING
   cfg=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=a.max_new_tokens,eos_token_id=t.eos_token_id,pad_token_id=t.pad_token_id,forced_eos_token_id=None)
   x=torch.tensor([ids],device=a.device)
   with torch.no_grad():seq=m.generate(input_ids=x,attention_mask=torch.ones_like(x),generation_config=cfg)[0].tolist()
   answer=seq[len(ids):];count=len(answer);truncated=not bool(answer and answer[-1]==t.eos_token_id);termination='length' if truncated else 'eos';completion=t.decode(answer,skip_special_tokens=True)
  results.append({k:row[k] for k in ['problem_id','group_id','family','stage','domain']}|{'completion':completion,'checkpoint_id':checkpoint_id,'engine':a.engine,'generated_tokens':count,'truncated':truncated,'termination':termination,'seconds':time.perf_counter()-start})
  if (i+1)%16==0 or i+1==len(rows):print(f'inference {i+1}/{len(rows)} engine={a.engine}',flush=True)
 write_jsonl(destination,results);write_json(str(destination)+'.manifest.json',{'args':vars(a),'engine':a.engine,'n':len(results),'checkpoint_id':checkpoint_id,'base_identity':identity,'adapter_sha256':adapter_sha,'source_data_sha256':file_sha(a.data),'predictions_sha256':file_sha(destination),'teacher_models_loaded':0,'student_checkpoints_loaded':int(a.engine=='hf'),'non_model_reference':a.engine!='hf'})
if __name__=='__main__':main()
