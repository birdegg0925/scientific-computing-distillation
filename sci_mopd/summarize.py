from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from .common import write_json

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--evaluations',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args(argv);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for path in a.evaluations:
  s=json.loads(Path(path).read_text());rows.append({'evaluation':path,'n':s['n'],'analysis':s['by_stage']['analysis']['rate'],'code':s['by_stage']['code']['rate'],'primary_macro':s['primary_macro'],'anchor':s['anchor_rate'],'human_reasoning_reviewed':s['manual_reasoning_review']['reviewed']})
 write_json(out/'comparison.json',{'rows':rows,'note':'Oracle and rule scores describe reference and shortcut baselines.'})
 with (out/'comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(f'summarized {len(rows)} evaluations')
if __name__=='__main__':main()
