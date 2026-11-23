"""Shared tokenizer/base, adapter manifests, pure sampling, and assistant-only batches."""
from __future__ import annotations
import json
from pathlib import Path
from .common import model_messages,object_sha,file_sha,write_json
BASE_ID='Qwen/Qwen3-1.7B'
BASE_REVISION='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
def library_versions():
 import importlib.metadata
 return {k:importlib.metadata.version(k) for k in ['numpy','torch','transformers','peft','accelerate']}
def load_tokenizer(model):
 from transformers import AutoTokenizer
 t=AutoTokenizer.from_pretrained(model,revision=BASE_REVISION if model==BASE_ID else None)
 if t.pad_token_id is None:t.pad_token=t.eos_token
 t.padding_side='right';return t

def base_identity(model,tokenizer):
 path=Path(model)
 if path.is_dir():
  weights=sorted(list(path.glob('*.safetensors'))+list(path.glob('pytorch_model*.bin')))
  source={'local_weights':{p.name:file_sha(p) for p in weights},'config':file_sha(path/'config.json')}
 else:source={'model_id':model,'revision':BASE_REVISION if model==BASE_ID else 'UNPINNED_CUSTOM_MODEL'}
 return {'source':source,'tokenizer_sha256':object_sha(tokenizer.get_vocab())}
def load_model(model,adapter=None,trainable=False,rank=4,device='cpu',identity=None):
 import torch
 from transformers import AutoModelForCausalLM
 from peft import PeftModel,LoraConfig,TaskType,get_peft_model
 dtype=torch.float32 if not str(device).startswith('cuda') else torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
 m=AutoModelForCausalLM.from_pretrained(model,revision=BASE_REVISION if model==BASE_ID else None,torch_dtype=dtype).to(device)
 if adapter:
  manifest=Path(adapter)/'adapter_manifest.json'
  if identity is not None:
   if not manifest.exists() or json.loads(manifest.read_text())['base_identity']!=identity:raise ValueError('teacher/student adapter base or tokenizer identity mismatch')
  m=PeftModel.from_pretrained(m,adapter,is_trainable=trainable)
 elif trainable:m=get_peft_model(m,LoraConfig(task_type=TaskType.CAUSAL_LM,target_modules=['q_proj','v_proj'],r=rank,lora_alpha=2*rank,lora_dropout=0.0,bias='none'))
 m.eval();return m

def prompt_ids(t,row):return t.apply_chat_template(model_messages(row),tokenize=True,add_generation_prompt=True,enable_thinking=False)

def pack_sequences(sequences,prompt_lengths,pad_id,device='cpu'):
 import torch
 from .losses import assistant_mask
 n=max(map(len,sequences));ids=torch.tensor([s+[pad_id]*(n-len(s)) for s in sequences],device=device)
 lengths=torch.tensor([len(s) for s in sequences],device=device);prompts=torch.tensor(prompt_lengths,device=device)
 attention=torch.arange(n,device=device)[None]<lengths[:,None]
 mask=assistant_mask(ids,prompts,lengths)
 return ids,attention.long(),mask

def sft_batch(t,rows,max_length,device='cpu'):
 sequences=[];lengths=[]
 for row in rows:
  p=prompt_ids(t,row);answer=t(row['target'],add_special_tokens=False)['input_ids']+[t.eos_token_id]
  if len(p)+len(answer)>max_length:raise ValueError(f'full SFT example exceeds max_length={max_length}; increase it, no truncation is performed')
  sequences.append(p+answer);lengths.append(len(p))
 ids,attention,mask=pack_sequences(sequences,lengths,t.pad_token_id,device);return ids,attention,mask

def sample_fresh(model,t,rows,max_prompt_tokens,max_new_tokens,device='cpu'):
 import torch
 from transformers import GenerationConfig
 from .losses import SAMPLING
 model.eval();records=[]
 config=GenerationConfig(**SAMPLING,max_new_tokens=max_new_tokens,min_new_tokens=0,bos_token_id=t.bos_token_id if t.bos_token_id is not None else model.config.bos_token_id,eos_token_id=t.eos_token_id,pad_token_id=t.pad_token_id,
                         forced_bos_token_id=None,forced_eos_token_id=None,bad_words_ids=None,suppress_tokens=None,begin_suppress_tokens=None)
 if max_new_tokens<1:raise ValueError('max_new_tokens must be positive')
 for row in rows:
  p=prompt_ids(t,row)
  if len(p)>max_prompt_tokens:raise ValueError('prompt exceeds max_prompt_tokens; no silent truncation')
  x=torch.tensor([p],device=device)
  with torch.no_grad():y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),generation_config=config)[0].tolist()
  completion=y[len(p):];eos=bool(completion and completion[-1]==t.eos_token_id)
  if not completion:raise ValueError('empty sampled prefix; domain cannot be silently dropped')
  records.append({'row':row,'sequence':y,'prompt_length':len(p),'completion_ids':completion,'completion':t.decode(completion,skip_special_tokens=True),
                  'truncated':not eos,'termination':'eos' if eos else 'length','generated_tokens':len(completion),'sampling':SAMPLING.copy()})
 ids,attention,mask=pack_sequences([r['sequence'] for r in records],[r['prompt_length'] for r in records],t.pad_token_id,device)
 return records,ids,attention,mask

def save_student(model,tokenizer,out,identity,metadata):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);model.save_pretrained(out);tokenizer.save_pretrained(out)
 write_json(out/'adapter_manifest.json',{'base_identity':identity,'metadata':metadata,'deployment':'single student checkpoint; no teacher routing needed at inference'})
