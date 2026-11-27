"""Read-only local provenance audit; no model inference or training."""
import os,json,hashlib,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT)); import numpy as np
records=[]; skipped=[]; files=[]; seeds=set(); errors=[]
robot_keys={'point_mm','target_robot_mm','reachable_target_robot_mm','nominal_target_robot_mm'}
world_keys={'target_world_mm','reachable_target_world_mm','nominal_target_world_mm'}
def walk(v,source,loc=''):
 if isinstance(v,dict):
  for k,x in v.items():
   if k in robot_keys|world_keys and isinstance(x,list) and len(x)==3 and all(isinstance(a,(int,float)) for a in x):
    records.append(dict(point=x,frame='robot' if k in robot_keys else 'world',source=source,key=loc+'/'+k))
   if 'seed' in k.lower():
    if isinstance(x,int): seeds.add(x)
    elif isinstance(x,list): seeds.update(a for a in x if isinstance(a,int))
   if isinstance(x,(dict,list)): walk(x,source,loc+'/'+k)
 elif isinstance(v,list):
  for i,x in enumerate(v):
   if isinstance(x,(dict,list)):walk(x,source,loc+'/'+str(i))
for base in ['configs','runs','outputs','data','github_reports']:
 for folder,dirs,names in os.walk(ROOT/base):
  dirs[:]=[d for d in dirs if d not in {'.git','__pycache__','.venv'}]
  if Path(folder).is_relative_to(OUT):dirs[:]=[];continue
  for n in names:
   p=Path(folder)/n
   if p.suffix not in {'.json','.jsonl'}:continue
   rel=str(p.relative_to(ROOT));size=p.stat().st_size
   # Large step streams repeat the target; inspect first and last records.
   try:
    if p.suffix=='.json' and size<=30_000_000:
     data=p.read_bytes();walk(json.loads(data),rel);files.append(dict(path=rel,size=size,sha256=hashlib.sha256(data).hexdigest(),coverage='whole_json'))
    elif p.suffix=='.jsonl':
     with p.open('rb') as f:
      first=f.readline();f.seek(max(0,size-262144));tail=f.read().splitlines()
     lines=[first]+tail[-2:]
     for i,line in enumerate(lines):
      try:walk(json.loads(line),rel,'sample'+str(i))
      except Exception:pass
     files.append(dict(path=rel,size=size,coverage='first_and_last_two_jsonl_records'))
    else:skipped.append(dict(path=rel,size=size,reason='json_above_30MB'))
   except Exception as e:errors.append(dict(path=rel,error=str(e)))
(OUT/'history_points.json').write_text(json.dumps(records))
(OUT/'history_inventory.json').write_text(json.dumps(dict(files=files,skipped=skipped,errors=errors,seeds=sorted(seeds),point_count=len(records),limits=['JSONL sampled at boundaries; binary datasets/checkpoints not decoded by provenance audit','Local checkout only; absence is not proof of absence in external history']),indent=2))
m=json.load(open(ROOT/'configs/target_split_manifest.json')); summary={}
for key in ['training_targets','dagger_targets','development_targets','final_test_targets']:
 rows=[]
 for t in m[key]:
  p=np.array(t['point_mm']);matches=[]
  for r in records:
   if r['frame']=='robot' and np.linalg.norm(p-r['point'])<=.001:
    source=r['source']
    if source.startswith(('outputs/','data/')) and not ('manifest' in source or '/code/' in source):matches.append(source)
  rows.append(dict(id=t['id'],point_mm=t['point_mm'],declared_roles=t['roles'],declared_sources=t['sources'],historical_outputs_or_data_matches=sorted(set(matches))))
 summary[key]=rows
(OUT/'existing_target_usage.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(dict(files=len(files),points=len(records),seeds=len(seeds),skipped=len(skipped),errors=len(errors),splits={k:dict(count=len(v),with_historical_output_data_match=sum(bool(t['historical_outputs_or_data_matches']) for t in v)) for k,v in summary.items()})))
