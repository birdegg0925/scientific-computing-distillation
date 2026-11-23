"""Synchronous fresh-rollout, stage-routed, immediate reverse-KL proxy distillation."""
from __future__ import annotations
import argparse,random,time,json
from pathlib import Path
from .common import read_jsonl,write_json,write_jsonl,file_sha
from .hf import BASE_ID,load_tokenizer,base_identity,load_model,sample_fresh,save_student,library_versions

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--teacher-math',required=True);p.add_argument('--teacher-code',required=True);p.add_argument('--out',required=True);p.add_argument('--model',default=BASE_ID);p.add_argument('--student-init')
 p.add_argument('--steps',type=int,default=20);p.add_argument('--max-prompt-tokens',type=int,default=2048);p.add_argument('--max-new-tokens',type=int,default=256);p.add_argument('--samples-per-domain',type=int,default=1)
 p.add_argument('--g-analysis',type=float,default=.4);p.add_argument('--g-code',type=float,default=.4);p.add_argument('--g-anchor',type=float,default=.2);p.add_argument('--adv-clip',type=float,default=10.)
 p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--seed',type=int,default=7);p.add_argument('--device',default='cpu');p.add_argument('--teacher-device',default='cpu');p.add_argument('--rank',type=int,default=4);a=p.parse_args(argv)
 if a.steps<1:raise ValueError('steps must be positive')
 import torch
 from .losses import freeze,token_logprobs,weighted_proxy
 torch.manual_seed(a.seed);random.seed(a.seed);torch.set_num_threads(2)
 g={'analysis':a.g_analysis,'code':a.g_code,'anchor':a.g_anchor}
 if any(v<0 for v in g.values()) or abs(sum(g.values())-1)>1e-7 or a.samples_per_domain<1:raise ValueError('fixed g must be nonnegative and sum to one; at least one sample per domain')
 rows=read_jsonl(a.data);pools={d:[r for r in rows if r['split']=='train' and r['domain']==d and r['pool']==('anchor' if d=='anchor' else 'distill')] for d in g}
 if any(not rs for rs in pools.values()):raise ValueError('all three training domains must be present')
 t=load_tokenizer(a.model);identity=base_identity(a.model,t);student=load_model(a.model,a.student_init,True,a.rank,a.device,identity)
 teachers={'analysis':freeze(load_model(a.model,a.teacher_math,device=a.teacher_device,identity=identity)),
           'code':freeze(load_model(a.model,a.teacher_code,device=a.teacher_device,identity=identity)),
           'anchor':freeze(load_model(a.model,a.student_init,device=a.device,identity=identity))}
 for d,path in [('analysis',a.teacher_math),('code',a.teacher_code)]:
  meta=json.loads((Path(path)/'adapter_manifest.json').read_text())['metadata']
  if meta.get('stage')!=d:raise ValueError('specialist teacher stage manifest mismatch: '+d)
 opt=torch.optim.AdamW([q for q in student.parameters() if q.requires_grad],lr=a.lr);rng=random.Random(a.seed);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 logs=[];rollout_rows=[];start=time.perf_counter()
 for step in range(a.steps):
  packs={};domain_records={}
  # No parameter update occurs until every domain's fresh sample and scores are collected.
  for d in g:
   chosen=[rng.choice(pools[d]) for _ in range(a.samples_per_domain)]
   records,ids,att,mask=sample_fresh(student,t,chosen,a.max_prompt_tokens,a.max_new_tokens,a.device)
   student.eval()
   with torch.no_grad():
    old=token_logprobs(student(input_ids=ids,attention_mask=att,use_cache=False).logits,ids).detach()
    teacher_device=a.device if d=='anchor' else a.teacher_device
    ti,ta=ids.to(teacher_device),att.to(teacher_device)
    target=token_logprobs(teachers[d](input_ids=ti,attention_mask=ta,use_cache=False).logits,ti).detach().to(a.device)
   cur=token_logprobs(student(input_ids=ids,attention_mask=att,use_cache=False).logits,ids)
   packs[d]=(cur,old,target,mask);domain_records[d]=records
  loss,metrics=weighted_proxy(packs,g,a.adv_clip);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([q for q in student.parameters() if q.requires_grad],1.);opt.step()
  for d,records in domain_records.items():
   metrics[d]['truncated_samples']=sum(r['truncated'] for r in records)
   for ri,r in enumerate(records):
    old,teacher,mask=packs[d][1],packs[d][2],packs[d][3]
    rollout_rows.append({'step':step+1,'domain':d,'problem_id':r['row']['problem_id'],'completion':r['completion'],'completion_ids':r['completion_ids'],'assistant_score_positions':mask[ri].nonzero().flatten().detach().cpu().tolist(),'old_logprobs':old[ri][mask[ri]].detach().cpu().tolist(),'teacher_logprobs':teacher[ri][mask[ri]].detach().cpu().tolist(),'truncated':r['truncated'],'termination':r['termination'],'sampling':r['sampling']})
  logs.append({'step':step+1,'loss':float(loss.detach()),'domains':metrics});print(f'mopd step={step+1}/{a.steps} proxy={float(loss.detach()):.6f} truncated='+str(sum(metrics[d]['truncated_samples'] for d in g)),flush=True)
 metadata={'kind':'immediate per-token reverse-KL score proxy, not full-sequence unbiased MOPD reproduction','teacher_routes':{'analysis':a.teacher_math,'code':a.teacher_code,'anchor':'frozen exact student start'},
 'runtime_dtypes':{'student':str(next(student.parameters()).dtype),**{d:str(next(tmodel.parameters()).dtype) for d,tmodel in teachers.items()}},'student_start_adapter_hashes':{f.name:file_sha(f) for f in Path(a.student_init).glob('*') if f.name in ['adapter_model.safetensors','adapter_config.json','adapter_manifest.json']} if a.student_init else {},'student_start':'common instruction base' if not a.student_init else a.student_init,'input_data_sha256':file_sha(a.data),'teacher_artifact_hashes':{d:{f.name:file_sha(f) for f in Path(path).glob('*') if f.name in ['adapter_model.safetensors','adapter_manifest.json']} for d,path in [('analysis',a.teacher_math),('code',a.teacher_code)]},'zero_weight_domains_still_sampled':[d for d,v in g.items() if v==0],'fixed_g':g,'fresh_rollout_updates':1,'dropout_disabled_for_rollout_and_scoring':True,'all_sampled_prefixes_retained':True,'args':vars(a)}
 save_student(student,t,out,identity,metadata);write_json(out/'mopd_run.json',{'metadata':metadata,'versions':library_versions(),'seconds':time.perf_counter()-start,'logs':logs,'execution_device':a.device,'teacher_device':a.teacher_device,'anchor_device':a.device,'memory_strategy':'two specialist teachers resident on teacher-device; student and frozen anchor on student device'});write_jsonl(out/'student_rollouts.jsonl',rollout_rows)
if __name__=='__main__':main()
