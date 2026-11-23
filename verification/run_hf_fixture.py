"""Local CPU pipeline check with a randomly initialized miniature Transformers model and PEFT adapters."""
import subprocess,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sci_mopd.common import read_jsonl,write_jsonl,write_json
rows=read_jsonl('data_pilot/test.jsonl');small=[next(r for r in rows if r['stage']==s) for s in ['analysis','code','anchor']];write_jsonl('verification/hf_fixture_inputs.jsonl',small)
commands=[['sft','--data','data_pilot/train.jsonl','--out','verification/teacher_math','--model','verification/random_hf_model','--stage','analysis','--pool','teacher_sft','--steps','1'],
['sft','--data','data_pilot/train.jsonl','--out','verification/teacher_code','--model','verification/random_hf_model','--stage','code','--pool','teacher_sft','--steps','1'],
['train','--data','data_pilot/train.jsonl','--teacher-math','verification/teacher_math','--teacher-code','verification/teacher_code','--out','verification/student','--model','verification/random_hf_model','--steps','1','--max-new-tokens','4'],
['infer','--data','verification/hf_fixture_inputs.jsonl','--out','verification/hf_fixture_predictions.jsonl','--engine','hf','--model','verification/random_hf_model','--adapter','verification/student','--max-new-tokens','4'],
['evaluate','--data','verification/hf_fixture_inputs.jsonl','--predictions','verification/hf_fixture_predictions.jsonl','--out','verification/hf_fixture_evaluation']]
scope = 'Local CPU pipeline check with a randomly initialized miniature Transformers model and PEFT adapters.'
status=[]
for i,args in enumerate(commands,1):
 with open(f'verification/hf_phase_{i}.log','w') as f:r=subprocess.run([sys.executable,'-m','sci_mopd.cli',*args],stdout=f,stderr=subprocess.STDOUT)
 status.append({'phase':i,'command':['python','-m','sci_mopd.cli',*args],'exit_code':r.returncode})
 if r.returncode:write_json('verification/hf_fixture_status.json',{'scope':scope,'phases':status});raise SystemExit(r.returncode)
write_json('verification/hf_fixture_status.json',{'scope':scope,'phases':status,'all_exit_zero':True})
print('HF/PEFT local tiny fixture phases passed:',len(status))
