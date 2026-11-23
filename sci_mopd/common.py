from __future__ import annotations
import hashlib,json
from pathlib import Path
STAGES=('analysis','code','anchor')
CONDITIONS=('not_applicable','well_conditioned','ill_conditioned','singular')
# Task version 2: the first four non-ok issues can occur in a derivation, the last four in a program.
ISSUES=('ok','antiderivative','arithmetic','integration_sign','rhs_scale','singular_system','simpson_weights')
DERIVATION_ISSUES=('ok','antiderivative','arithmetic','integration_sign','rhs_scale','singular_system')
PROGRAM_ISSUES=('ok','integration_sign','rhs_scale','singular_system','simpson_weights')
def write_json(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def write_jsonl(path,rows):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
def read_jsonl(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def object_sha(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def model_messages(row):
 # Gold/reference/public metadata are never appended to model input.
 return [{'role':'system','content':'Solve the scientific-computing task. Follow the requested output format. State uncertainty instead of inventing evidence.'},
         {'role':'user','content':row['prompt']}]
def strip_code(text):
 text=text.strip()
 if text.startswith('```'):
  lines=text.splitlines()
  if len(lines)<3 or lines[-1].strip()!='```' or sum(x.strip().startswith('```') for x in lines)!=2:raise ValueError('one complete code block required')
  if lines[0].strip() not in ['```python','```py','```']:raise ValueError('unsupported code fence')
  return '\n'.join(lines[1:-1])
 return text
