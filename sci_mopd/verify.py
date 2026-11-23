"""Bounded numeric AST interpreter. It is deliberately not an arbitrary Python sandbox."""
from __future__ import annotations
import ast,json,math,operator,re,copy
import numpy as np
from .common import strip_code

class RejectedCode(ValueError):pass
class NumericInterpreter:
 def __init__(self,max_nodes=20000,max_array=4096,max_loop=1024):
  self.max_nodes=max_nodes;self.max_array=max_array;self.max_loop=max_loop;self.visits=0;self.env={}
  self.calls={'np.array':np.array,'np.linspace':self.linspace,'np.arange':self.arange,'np.zeros':np.zeros,'np.ones':np.ones,'np.eye':np.eye,
   'np.sum':np.sum,'np.dot':self.dot,'np.abs':np.abs,'np.max':np.max,'np.min':np.min,'np.sin':np.sin,'np.cos':np.cos,'np.exp':np.exp,
   'np.linalg.solve':np.linalg.solve,'np.linalg.norm':np.linalg.norm,'np.linalg.det':np.linalg.det,'np.linalg.cond':np.linalg.cond,'np.linalg.matrix_rank':np.linalg.matrix_rank,
   'math.sin':math.sin,'math.cos':math.cos,'math.exp':math.exp,'abs':abs,'sum':sum,'len':len,'float':float,'int':int,'round':round,'range':self.bounded_range}
 def tick(self):
  self.visits+=1
  if self.visits>self.max_nodes:raise RejectedCode('AST operation budget exceeded')
 def bounded_range(self,*args):
  r=range(*args)
  if len(r)>self.max_loop:raise RejectedCode('range exceeds loop limit')
  return r
 def linspace(self,a,b,num=50,**kwargs):
  if not isinstance(num,(int,np.integer)) or not 1<=num<=self.max_array:raise RejectedCode('linspace size limit')
  return np.linspace(a,b,num,**kwargs)
 def arange(self,*args,**kwargs):
  if any(not np.isscalar(x) for x in args):raise RejectedCode('arange scalar arguments required')
  start,stop,step=(0,args[0],1) if len(args)==1 else (args[0],args[1],1) if len(args)==2 else args
  if step==0 or abs((stop-start)/step)>self.max_array:raise RejectedCode('arange size limit')
  return np.arange(*args,**kwargs)
 def dot(self,a,b):
  sa,sb=np.shape(a),np.shape(b)
  if len(sa)>2 or len(sb)>2:raise RejectedCode('dot/matmul limited to vectors and matrices')
  shape=sb if not sa else sa if not sb else () if len(sa)==len(sb)==1 else (sa[0],) if len(sb)==1 else (sb[-1],) if len(sa)==1 else (sa[0],sb[-1])
  if math.prod(shape)>self.max_array:raise RejectedCode('dot output size limit before allocation')
  return np.dot(a,b)
 def element_count(self,x,depth=0):
  if depth>16:raise RejectedCode('nested container depth limit')
  if isinstance(x,np.ndarray):return x.size
  if not isinstance(x,(list,tuple)):return 1
  n=0
  for v in x:
   n+=self.element_count(v,depth+1)
   if n>self.max_array:raise RejectedCode('array conversion size limit before allocation')
  return n
 def checked(self,x):
  if isinstance(x,np.ndarray):
   if x.size>self.max_array or x.dtype.kind not in 'bifu':raise RejectedCode('numeric array shape/type limit')
   if not np.isfinite(x).all() or (x.size and np.max(np.abs(x))>1e12):raise RejectedCode('nonfinite or excessive numeric value')
  elif isinstance(x,(int,float,np.number)) and (not math.isfinite(float(x)) or abs(float(x))>1e12):raise RejectedCode('nonfinite or excessive scalar')
  elif isinstance(x,(list,tuple)) and len(x)>self.max_array:raise RejectedCode('container size limit')
  return x
 def dotted(self,node):
  if isinstance(node,ast.Name):return node.id
  if isinstance(node,ast.Attribute):
   if node.attr.startswith('_'):raise RejectedCode('private attributes forbidden')
   return self.dotted(node.value)+'.'+node.attr
  raise RejectedCode('dynamic attribute access forbidden')
 def expr(self,n):
  self.tick()
  if isinstance(n,ast.Constant):
   if isinstance(n.value,(bool,int,float,str)) or n.value is None:return self.checked(n.value)
   raise RejectedCode('constant type forbidden')
  if isinstance(n,ast.Name):
   if n.id.startswith('_'):raise RejectedCode('private name forbidden')
   if n.id in self.env:return self.env[n.id]
   if n.id in ['float','int']:return {'float':float,'int':int}[n.id]
   raise RejectedCode('unknown name '+n.id)
  if isinstance(n,(ast.List,ast.Tuple)):return self.checked([self.expr(x) for x in n.elts])
  if isinstance(n,ast.UnaryOp):
   op={ast.USub:operator.neg,ast.UAdd:operator.pos,ast.Not:operator.not_}.get(type(n.op))
   if op is None:raise RejectedCode('unary operator forbidden')
   return self.checked(op(self.expr(n.operand)))
  if isinstance(n,ast.BinOp):
   ops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.Pow:operator.pow,ast.Mod:operator.mod,ast.MatMult:operator.matmul}
   if type(n.op) not in ops:raise RejectedCode('binary operator forbidden')
   a,b=self.expr(n.left),self.expr(n.right)
   if isinstance(a,(list,tuple,str)) or isinstance(b,(list,tuple,str)):
    if isinstance(n.op,ast.Mult):raise RejectedCode('container repetition forbidden before allocation')
    if isinstance(n.op,ast.Add):
     if type(a)!=type(b) or len(a)+len(b)>self.max_array:raise RejectedCode('container concatenation size/type limit')
    else:raise RejectedCode('container arithmetic unsupported')
   if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
    if isinstance(n.op,ast.MatMult):return self.checked(self.dot(a,b))
    try:shape=np.broadcast_shapes(np.shape(a),np.shape(b))
    except ValueError:raise RejectedCode('broadcast shape mismatch')
    if math.prod(shape)>self.max_array:raise RejectedCode('broadcast result size limit before allocation')
   if isinstance(n.op,ast.Pow) and (not np.isscalar(b) or abs(float(b))>12):raise RejectedCode('power exponent limit')
   return self.checked(ops[type(n.op)](a,b))
  if isinstance(n,ast.Subscript):return self.checked(copy.deepcopy(self.expr(n.value)[self.slice(n.slice)]))
  if isinstance(n,ast.Attribute):
   d=self.dotted(n)
   if d in ['math.pi','np.pi']:return math.pi
   if isinstance(n.value,ast.Name) and n.attr=='T' and isinstance(self.env.get(n.value.id),np.ndarray):return self.env[n.value.id].T
   raise RejectedCode('attribute not allowed')
  if isinstance(n,ast.Call):
   name=self.dotted(n.func)
   if name not in self.calls:raise RejectedCode('call not allowed: '+name)
   if any(k.arg is None for k in n.keywords):raise RejectedCode('expanded keyword arguments forbidden')
   kwargs={k.arg:self.expr(k.value) for k in n.keywords}
   if set(kwargs)-{'dtype','axis','num','endpoint'}:raise RejectedCode('keyword not allowed')
   args=[self.expr(x) for x in n.args]
   if name=='np.array' and args:self.element_count(args[0])
   if name in ['np.zeros','np.ones','np.eye'] and args:
    shape=(args[:2] if len(args)>1 else [args[0],args[0]]) if name=='np.eye' else args[0] if isinstance(args[0],(list,tuple)) else [args[0]]
    if any(not isinstance(x,(int,np.integer)) or x<0 for x in shape) or math.prod(int(x) for x in shape)>self.max_array:raise RejectedCode('allocation shape/size limit before allocation')
   return self.checked(self.calls[name](*args,**kwargs))
  if isinstance(n,ast.Compare):
   ops={ast.Lt:operator.lt,ast.LtE:operator.le,ast.Gt:operator.gt,ast.GtE:operator.ge,ast.Eq:operator.eq,ast.NotEq:operator.ne}
   left=self.expr(n.left)
   for op,right in zip(n.ops,n.comparators):
    if type(op) not in ops:raise RejectedCode('comparison forbidden')
    right=self.expr(right)
    if not bool(ops[type(op)](left,right)):return False
    left=right
   return True
  if isinstance(n,ast.BoolOp):
   for child in n.values[:-1]:
    value=self.expr(child)
    if isinstance(n.op,ast.And) and not bool(value):return value
    if isinstance(n.op,ast.Or) and bool(value):return value
   return self.expr(n.values[-1])
  raise RejectedCode('expression not allowed: '+type(n).__name__)
 def slice(self,n):
  if isinstance(n,ast.Slice):return slice(*(self.expr(x) if x is not None else None for x in [n.lower,n.upper,n.step]))
  if isinstance(n,ast.Tuple):return tuple(self.slice(x) for x in n.elts)
  return self.expr(n)
 def assign(self,target,value):
  if isinstance(target,ast.Name):
   if target.id.startswith('_') or target.id in ['p','np','math','float','int','range','open','exec','eval']:raise RejectedCode('protected assignment')
   if isinstance(value,dict):raise RejectedCode('input dictionary cannot be aliased')
   self.env[target.id]=self.checked(value);return
  if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id!='p':self.env[target.value.id][self.slice(target.slice)]=value;return
  raise RejectedCode('assignment target forbidden')
 def stmt(self,n):
  self.tick()
  if isinstance(n,ast.Import):
   if len(n.names)!=1 or (n.names[0].name,n.names[0].asname) not in [('numpy','np'),('math',None)]:raise RejectedCode('only import numpy as np / import math')
  elif isinstance(n,ast.Assign):
   if len(n.targets)!=1:raise RejectedCode('one assignment target required')
   self.assign(n.targets[0],self.expr(n.value))
  elif isinstance(n,ast.AugAssign):
   if not isinstance(n.target,ast.Name):raise RejectedCode('augmented target must be name')
   self.assign(n.target,self.expr(ast.BinOp(left=n.target,op=n.op,right=n.value)))
  elif isinstance(n,ast.If):
   for s in n.body if bool(self.expr(n.test)) else n.orelse:self.stmt(s)
  elif isinstance(n,ast.For):
   values=self.expr(n.iter)
   if not isinstance(values,(range,list,tuple)) or len(values)>self.max_loop:raise RejectedCode('bounded iteration required')
   for v in values:
    self.assign(n.target,v)
    for s in n.body:self.stmt(s)
   if n.orelse:raise RejectedCode('for-else unsupported')
  elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):pass
  else:raise RejectedCode('statement not allowed: '+type(n).__name__)
 def run(self,text,problem=None):
  if problem is not None:self.env['p']=copy.deepcopy(problem)
  tree=ast.parse(strip_code(text))
  if len(list(ast.walk(tree)))>2000:raise RejectedCode('AST source size limit')
  for n in tree.body:self.stmt(n)
  if 'result' not in self.env:raise RejectedCode('result variable required')
  return self.env

def final_record(text):
 if text.count('FINAL=')!=1:raise ValueError('exactly one FINAL= record required')
 reason,tail=text.split('FINAL=',1);obj,end=json.JSONDecoder().raw_decode(tail.strip())
 if tail.strip()[end:].strip():raise ValueError('text after final JSON record')
 if not isinstance(obj,dict) or not {'value','condition','verdict','issue'}<=set(obj):raise ValueError('FINAL requires value, condition, verdict, issue')
 return obj,bool(reason.strip())
def numeric_check(value,row):
 r=row['private_reference'];target=r['value']
 if target is None:return value is None,{'residual':None}
 try:
  def numeric_types(x):
   if isinstance(x,(list,tuple)):return all(numeric_types(y) for y in x)
   if isinstance(x,np.ndarray):return x.dtype.kind in 'fiu'
   return isinstance(x,(int,float,np.number)) and not isinstance(x,(bool,np.bool_))
  if not numeric_types(value):return False,{'residual':None}
  a=np.asarray(value,dtype=float);b=np.asarray(target,dtype=float)
  if a.shape!=b.shape or not np.isfinite(a).all():return False,{'residual':None}
  ok=bool(np.allclose(a,b,atol=r['atol'],rtol=r['rtol']));res=None
  if row['family']=='linear':
   A=np.asarray(row['public_problem']['A']);rhs=np.asarray(row['public_problem']['b']);res=float(np.linalg.norm(A@a-rhs)/(np.linalg.norm(A)*np.linalg.norm(a)+np.linalg.norm(rhs)+1e-12));ok=ok and res<=r['residual_tolerance']
  return ok,{'residual':res}
 except (ValueError,TypeError,OverflowError):return False,{'residual':None}
def hidden_rebindings(row):
 p=row['public_problem'];probes=[]
 for k in [1,2]:
  q=copy.deepcopy(p)
  if p['kind']=='integral':
   q['a']=p['a']+.07*k;q['b']=p['b']+.11*k;q['coefficients']=[float(v)*(1+.13*k) for v in p['coefficients']]
  else:
   A=np.asarray(p['A']);b=np.asarray(p['b']);q['A']=(A*(1+.17*k)+(np.eye(len(b))*.037 if k==2 else 0)).tolist();q['b']=(b*(1+.23*k)+(.1*k if k==2 or np.linalg.matrix_rank(A)==len(b) else 0)).tolist()
  probes.append(q)
 return probes

def check_completion(row,text):
 base={'problem_id':row['problem_id'],'group_id':row['group_id'],'family':row['family'],'stage':row['stage'],'auto_pass':False,'reason_present':None,'manual_reasoning_review_required':row['stage']=='analysis'}
 try:
  if row['stage']=='anchor':
   obj=json.loads(text.strip());base['auto_pass']=isinstance(obj,dict) and set(obj)=={'label','count'} and isinstance(obj.get('label'),str) and type(obj.get('count')) is int and obj==row['private_reference']['expected_json'];return base
  if row['stage']=='analysis':obj,present=final_record(text);base['reason_present']=present;value=obj.get('value');condition_ok=obj.get('condition')==row['private_reference']['condition']
  else:
   obj=NumericInterpreter().run(text,row['public_problem']);value=obj['result'];condition_ok=True
  numeric,extra=numeric_check(value,row);r=row['private_reference'];labels=obj.get('verdict')==r['expected_verdict'] and obj.get('issue')==r['expected_issue']
  probe_passes=[]
  if row['stage']=='code':
   from .data import reference
   for hidden in hidden_rebindings(row):
    value2=NumericInterpreter().run(text,hidden)['result'];probe_row={**row,'public_problem':hidden,'private_reference':reference(hidden)}
    probe_passes.append(numeric_check(value2,probe_row)[0])
  base.update(numeric_ok=numeric,condition_ok=condition_ok,label_ok=labels,hidden_probe_passes=probe_passes,auto_pass=numeric and condition_ok and labels and all(probe_passes),**extra)
 except Exception as e:base['error']=type(e).__name__+': '+str(e)[:160]
 return base
