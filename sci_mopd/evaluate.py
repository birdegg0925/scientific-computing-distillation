"""Automatic terminal/program checks, paired problem-group bootstrap, and separate anchor retention."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from .common import read_jsonl,write_json,write_jsonl,file_sha
from .verify import check_completion

def evaluate_rows(data,predictions):
 data_ids=[r['problem_id'] for r in data]
 if len(data_ids)!=len(set(data_ids)):raise ValueError('duplicate data IDs')
 if set(r['problem_id'] for r in predictions)-set(data_ids):raise ValueError('prediction IDs outside data')
 pred={r['problem_id']:r for r in predictions}
 if len(pred)!=len(predictions):raise ValueError('duplicate prediction IDs')
 result=[]
 for r in data:
  q=pred.get(r['problem_id']);v=check_completion(r,q['completion']) if q else {'problem_id':r['problem_id'],'group_id':r['group_id'],'family':r['family'],'stage':r['stage'],'auto_pass':False,'error':'missing_prediction','manual_reasoning_review_required':r['stage']=='analysis'}
  v['engine']=q.get('engine') if q else None;v['truncated']=q.get('truncated') if q else None;result.append(v)
 return result

def macro(rows):
 rates=[sum(r['auto_pass'] for r in rows if r['stage']==s)/sum(r['stage']==s for r in rows) for s in ['analysis','code'] if any(r['stage']==s for r in rows)]
 return sum(rates)/len(rates) if len(rates)==2 else None

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--predictions',required=True);p.add_argument('--out',required=True);p.add_argument('--baseline-predictions');a=p.parse_args(argv)
 data=read_jsonl(a.data);predictions=read_jsonl(a.predictions);engines=set(r.get('engine') for r in predictions)
 if len(engines)!=1 or not engines<={'hf','oracle','accept_all','keyword'}:raise ValueError('one declared engine per evaluation is required')
 engine=next(iter(engines));checkpoint_ids=set(r.get('checkpoint_id') for r in predictions)
 if engine=='hf' and (len(checkpoint_ids)!=1 or not next(iter(checkpoint_ids))):raise ValueError('HF rows must share one nonempty checkpoint_id; per-stage peak mixtures are invalid')
 sidecar=Path(str(a.predictions)+'.manifest.json')
 if sidecar.exists():
  manifest=json.loads(sidecar.read_text())
  if manifest['predictions_sha256']!=file_sha(a.predictions) or manifest['source_data_sha256']!=file_sha(a.data):raise ValueError('inference manifest hash mismatch')
  if engine=='hf' and manifest['checkpoint_id']!=next(iter(checkpoint_ids)):raise ValueError('checkpoint identity differs from inference manifest')
 rows=evaluate_rows(data,predictions);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);write_jsonl(out/'rows.jsonl',rows)
 by={s:{'n':sum(r['stage']==s for r in rows),'auto_pass':sum(r['auto_pass'] and r['stage']==s for r in rows)} for s in ['analysis','code','anchor']}
 for s in by:by[s]['rate']=by[s]['auto_pass']/by[s]['n'] if by[s]['n'] else None
 summary={'engine':engine,'non_model_reference':engine!='hf','checkpoint_id':next(iter(checkpoint_ids)) if engine=='hf' else None,'n':len(rows),'by_stage':by,'primary_macro':macro(rows),'anchor_rate':by['anchor']['rate'],'manual_reasoning_review':{'required':by['analysis']['n'],'reviewed':0,'status':'not_performed'},'paired':None,
          'source_data_sha256':file_sha(a.data),'predictions_sha256':file_sha(a.predictions),'scope':'Automatic checks of numerical values, conditions, issue labels and restricted program execution.','code_numeric_instances_per_candidate':3}
 if a.baseline_predictions:
  base=evaluate_rows(data,read_jsonl(a.baseline_predictions));bm={r['problem_id']:r for r in base};initial_anchor=[r for r in rows if r['stage']=='anchor' and bm[r['problem_id']]['auto_pass']]
  groups=sorted(set(r['group_id'] for r in rows if r['stage']!='anchor'));rng=np.random.default_rng(177);differences=[]
  valid_primary=macro(rows) is not None and macro(base) is not None
  if valid_primary:
   for _ in range(2000):
    selected=rng.choice(groups,len(groups),replace=True);new=[];old=[]
    for group in selected:new.extend(r for r in rows if r['group_id']==group);old.extend(r for r in base if r['group_id']==group)
    differences.append(macro(new)-macro(old))
  summary['paired']={'primary_macro_difference':macro(rows)-macro(base) if valid_primary else None,'primary_pair_reason':None if valid_primary else 'both_primary_stages_required','problem_group_bootstrap_difference_95pct':[float(x) for x in np.quantile(differences,[.025,.975])] if valid_primary else None,
    'anchor_initial_correct_denominator':len(initial_anchor),'anchor_retained_numerator':sum(r['auto_pass'] for r in initial_anchor),
    'anchor_retention_rate':sum(r['auto_pass'] for r in initial_anchor)/len(initial_anchor) if initial_anchor else None,
    'recovery_ratio':None,'recovery_reason':'not computed without a separately admitted teacher and stable positive teacher-minus-start gap'}
 fields=['problem_id','group_id','family','stage','auto_pass','numeric_ok','condition_ok','label_ok','reason_present','manual_reasoning_review_required','residual','truncated','error']
 with (out/'results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 write_json(out/'summary.json',summary);print(summary)
if __name__=='__main__':main()
