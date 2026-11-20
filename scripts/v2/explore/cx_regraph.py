"""#138 A: the targeted re-extraction candidate (8,792 neurons), built from the raw MaleCNS v1.0
tables under v2_main's thresholds (edge >= 10 synapses, type pair >= 50 after the edge filter,
resolved transmitter sign, allowed superclasses, no KC/APL/DPM).

PYTHONPATH=python python scripts/v2/explore/cx_regraph.py RAW_DIR OUT_DIR   (run from the repo root)

Provenance: the external consultation's read-only pre-check (11-21), reconstructed verbatim from
its session; only the input/output paths and the final persistence block are the PM's. It keeps
every v2_main neuron and edge, adds the central-complex types PFNa, FB5AB, hDeltaC, FC2B, PFL3
(all eligible cells) with EPG, LNOa and LHCENT8, and per side and wind family (JO-C, JO-E) the
widest-bottleneck shortest path from the wind seeds to LNOa (first hop AMMC/SAD/WED, then
central-brain intrinsic cells; the right JO-C entry is the real right JO-CL 512238), and per side
the visual seed to EPG path. The induced edges among the selected cells are the raw edges that
pass the rule. Group bits of the new cells follow the paths (not the hop-bounded rule); the model
reads group_mask only for the turn/forward DNs. It then builds VO2 on the old and the candidate
graph and checks a forward/backward pass.
"""
import sys
from pathlib import Path
from collections import deque,Counter
import numpy as np, pandas as pd, yaml
from connectome_extract.tables import load_annotations,load_neurotransmitters
from connectome_extract.graph import assign_signs
from connectome_extract.rules import EdgeRule,collect_edges
from connectome_extract.download import FILES
from flytrack_core.contracts import GROUP_OLFACTORY,GROUP_WIND,GROUP_TURN,GROUP_FORWARD
cfg=yaml.safe_load(Path('configs/v2/extraction/v2_main.yaml').read_text()); col=cfg['columns'];raw=Path(sys.argv[1]);weight=raw/FILES['weights']
a=load_annotations(raw,col);types=a.set_index('id').type.to_dict()
z=np.load('data/subgraphs/v2_main/graph.npz');old=set(z['node_ids']);rawodor=set(z['node_ids'][z['input_mask']&GROUP_OLFACTORY!=0]);rawwind=set(z['node_ids'][z['input_mask']&GROUP_WIND!=0]);allmotor=set(z['node_ids'][z['group_mask']&(GROUP_TURN|GROUP_FORWARD)!=0]);sd=dict(zip(z['node_ids'],z['side']))
anchor=['PFNa','FB5AB','hDeltaC','FC2B','PFL3']
bridge=['LNOa','LHCENT8','FB2A','EPG','LHPV6q1','LAL138','SAD003','SAD004','SAD077','WED203']

import re
from flytrack_core.contracts import GROUP_VISUAL
ann=a.merge(load_neurotransmitters(raw,col),on='id',how='left',validate='one_to_one')
ann=assign_signs(ann,float(cfg['nt_confidence_min']))
elig=ann.sign.notna() & ann.type.ne('') & ann.superclass.isin(cfg['allowed_superclasses']) & ~ann['class'].isin(cfg['excluded_classes'])
for pat in cfg['excluded_type_patterns']: elig &= ~ann.type.str.fullmatch(pat)
allset=set(ann.loc[elig,'id']);cb=set(ann.loc[elig & ann.superclass.eq('cb_intrinsic'),'id'])
profiles=ann.set_index('id');side=profiles.side.to_dict();ty=profiles.type.to_dict()
ns={t:set(ann.loc[elig & ann.type.eq(t),'id']) for t in anchor+bridge}
print('selected anatomy',len(allset),'scope',len(cb),flush=True)
rule=EdgeRule(cfg,a,weight,col);es=collect_edges(weight,col,allset,rule);print('anatomical edges',len(es),flush=True)
selectedLNO=es[es.post.isin(ns['LNOa'])].copy();selectedLNO['t']=selectedLNO.pre.map(ty);print('LNOa in strong',selectedLNO.groupby('t').synapse_count.sum().sort_values(ascending=False).head(14).to_dict(),flush=True)
selectedPF=es[es.post.isin(ns['PFNa'])].copy();selectedPF['t']=selectedPF.pre.map(ty);print('PFNa in strong',selectedPF.groupby('t').synapse_count.sum().sort_values(ascending=False).head(14).to_dict(),flush=True)
base_sel=old.union(*(ns[t] for t in anchor+['EPG','LNOa','LHCENT8']))
ae=es[(es.post.isin(cb)) & (es.pre.isin(cb|rawwind|rawodor|set(z['node_ids'][z['input_mask']&GROUP_VISUAL!=0])))]
adj={}
for u,v,w in ae.itertuples(index=False,name=None):adj.setdefault(u,[]).append((v,int(w)))
def bestpaths(start,goal,forbid,firstpred=None,limit=8):
 prev=set(start);states={i:(10**7,(int(i),)) for i in start}
 found={}
 for hop in range(1,limit+1):
  nxt={}
  for u,(bott,path) in states.items():
   for v,w in adj.get(u,()):
    if v in prev or v in forbid or (hop==1 and firstpred and not firstpred(ty[v])):continue
    nb=min(bott,w); prior=nxt.get(v)
    if prior is None or nb>prior[0] or (nb==prior[0] and (*path,v)<prior[1]):nxt[v]=(nb,(*path,v))
  for v in nxt.keys()&goal:found[v]=nxt[v]
  prev|=nxt.keys()
  if goal<=found.keys():break
  states={v:x for v,x in nxt.items() if v not in goal}
  if not states:break
 return found
wv=lambda t:t.startswith(('AMMC','SAD','WED'))

mend=set().union(*(ns[t] for t in anchor),ns['EPG'])
root=(ann[col['root_side']].fillna('').astype(str).str.lower().map({'l':-1,'r':1,'left':-1,'right':1}))
validjo=elig & root.notna() & ann.type.str.fullmatch('(?:'+'|'.join(cfg['selection']['wind_seed_patterns'])+')')
alljo={s:set(ann.loc[validjo & root.eq(s),'id']) for s in (-1,1)}
# Rebuild search adjacency for all eligible wind seeds. After the first hop use only central intrinsic cells.
full_ae=es[es.post.isin(cb) & es.pre.isin(cb|alljo[-1]|alljo[1]|set(z['node_ids'][z['input_mask']&GROUP_VISUAL!=0]))]
adj={}
for u,v,w in full_ae[['pre','post','synapse_count']].itertuples(index=False,name=None):
 adj.setdefault(u,[]).append((v,int(w)))
extra=set();newseeds=set();dwind=[];allwindpaths=[]
for fam in ('JO-C','JO-E'):
 for s in (-1,1):
  start={u for u in rawwind if sd[u]==s and ty[u].startswith(fam)}
  found=bestpaths(start,ns['LNOa'],mend,wv)
  if ns['LNOa']-found.keys():
   add=bestpaths({u for u in alljo[s] if ty[u].startswith(fam)},ns['LNOa'],mend,wv)
   found.update({v:p for v,p in add.items() if v not in found})
  for u,(b,p) in sorted(found.items()):
   extra.update(p);allwindpaths.append((fam,s,side[u],p))
   if p[0] not in rawwind:newseeds.add(p[0])
   print('wind chosen',fam,s,side[u],b,[(ty[i],i) for i in p],flush=True)
  dwind.append((fam,s,len(found)))
visual=set(z['node_ids'][z['input_mask']&GROUP_VISUAL!=0]);vs=set();vlist=[]
for s in (-1,1):
 f=bestpaths({u for u in visual if sd[u]==s},ns['EPG'],mend-ns['EPG'])
 for targetside in (-1,1):
  choices=[(v,bb,pp) for v,(bb,pp) in f.items() if side[v]==targetside]
  if choices:
   v,bb,pp=min(choices,key=lambda k:(len(k[2]),-k[1],k[2]));vs.update(pp);vlist.append(pp)
   print('visual',s,targetside,len(choices),bb,[(ty[x],x) for x in pp],flush=True)
nodes=base_sel|extra|vs
e=es[es.pre.isin(nodes)&es.post.isin(nodes)].copy()
e['u']=e.pre.map(ty);e['v']=e.post.map(ty)
newset=nodes-old;outside=nodes-base_sel
print('FINAL',len(nodes),len(e),'new cells and edges',len(newset),len(e)-len(z['pre']),'new seeds',[(ty[q],q,int(root[ann.index[ann.id.eq(q)][0]])) for q in sorted(newseeds)],'wind_status',dwind,flush=True)
print('extra body outside anchors',[(ty[u],int(u)) for u in sorted(outside)],flush=True)
print('paircount',len(e[['u','v']].drop_duplicates()),'base',len(e[e.pre.isin(old)&e.post.isin(old)][['u','v']].drop_duplicates()),'types',len(set(e.u)|set(e.v)),len(z['type_names']))
from collections import defaultdict
nex=defaultdict(set);back=defaultdict(set)
for u,v in e[['pre','post']].itertuples(index=False,name=None):
 nex[u].add(v);back[v].add(u)
def walk(starts,adj):
 seen=set(starts); front=set(starts)
 while front:
  fut=set().union(*(adj[u] for u in front))-seen;seen.update(fut);front=fut
 return seen
sources=set(z['node_ids'][z['input_mask']!=0])|newseeds
full=walk(sources,nex)&walk(allmotor,back)
print('all nodes forward & reverse',len(full&nodes),len(nodes),'outside',[(ty[k],k) for k in nodes-full])
for fam,s,ss,p in allwindpaths:
 level={p[-1]}; nn=[1]
 for t in ['PFNa','hDeltaC','FC2B','PFL3','DNa02']:
  level={v for u in level for v in nex[u] if ty[v]==t};nn.append(len(level))
 print('wind allbody',fam,s,ss,nn,sorted((int(v),side[v]) for v in level))
for s in (-1,1):
 start={u for u in rawodor if sd[u]==s}; counts=[len(start)]
 level=start
 for cols0 in [('DM1_lPN','VA2_adPN'),('LHCENT8',),('FB5AB',),('hDeltaC',),('FC2B',),('PFL3',),('DNa02',)]:
  level={v for u in level for v in nex[u] if ty[v] in cols0};counts.append(len(level))
 print('odor allbody',s,counts,sorted((int(v),side[v]) for v in level))
arr_in=e.groupby('post').synapse_count.sum().to_dict();idx={u:i for i,u in enumerate(z['node_ids'])}
cova=[arr_in.get(u,0)/z['brain_input_synapses'][idx[u]] for u in sorted(allmotor)]
print('cov min',min(cova),'n motor',len(allmotor),'base',min(z['subgraph_input_synapses'][idx[u]]/z['brain_input_synapses'][idx[u]] for u in allmotor),flush=True)

# Construct runtime tensors in memory; reporting branch audit metadata is separate.
from connectome_extract.rules import brain_inputs
from flytrack_v2.model.graph import GraphV2
from flytrack_v2.model.config import ModelConfigV2
from flytrack_v2.model.policy import build_policy,parameter_table
from flytrack_v2.model.sensory import InputNorm
from flytrack_v2.contracts import SensorContractV2
from flytrack_v2.eval.runner import rotate_wind
from flytrack_core.contracts import GROUP_CONVERGENCE
import torch
torch.set_num_threads(1);torch.manual_seed(100)
pids=np.asarray(sorted(nodes));pos={int(u):i for i,u in enumerate(pids)};ss=profiles.loc[pids];names,ti=np.unique(ss.type.astype(str).to_numpy(dtype=str),return_inverse=True)
sides=ss.side.to_numpy(dtype=np.int8,copy=True);groups=np.zeros(len(pids),np.uint8);im=np.zeros(len(pids),np.uint8);protected=np.zeros(len(pids),np.uint8)
for j,u in enumerate(z['node_ids']):
 k=pos[int(u)];sides[k]=z['side'][j];groups[k]=z['group_mask'][j];im[k]=z['input_mask'][j];protected[k]=z['protected'][j]
for u in nodes-old:protected[pos[u]]=1
for u in newseeds:
 im[pos[u]]=GROUP_WIND;sides[pos[u]]=int(root[ann.index[ann.id.eq(u)][0]])
for fam,s,sside,path in allwindpaths:
 for u in path:groups[pos[u]]|=GROUP_WIND
for path in vlist:
 for u in path:groups[pos[u]]|=GROUP_VISUAL
for t in ('DM1_lPN','VA2_adPN','LHCENT8','FB5AB','hDeltaC','FC2B','PFL3'):
 for u in nodes:
  if ty[u]==t:groups[pos[u]]|=GROUP_OLFACTORY
for t in ('PFNa','hDeltaC','FC2B','PFL3'):
 for u in ns[t]:groups[pos[u]]|=GROUP_WIND
ngraw=brain_inputs(weight,col,newset)
brain=np.array([(z['brain_input_synapses'][idx[u]] if u in idx else ngraw[u]) for u in pids],dtype=np.float32)
eb=e.sort_values(['post','pre']);newg=GraphV2(node_ids=pids,type_names=names,type_index=ti.astype(np.int64),side=sides,sign=ss.sign.to_numpy(np.int8),pre=eb.pre.map(pos).to_numpy(np.int64),post=eb.post.map(pos).to_numpy(np.int64),synapse_count=eb.synapse_count.to_numpy(np.float32),input_mask=im,group_mask=groups,brain_input_synapses=brain,subgraph_input_synapses=np.array([arr_in.get(u,0) for u in pids],np.float32),protected=protected)
oldg=GraphV2.load(Path('data/subgraphs/v2_main/graph.npz'))
contract=SensorContractV2.from_json(Path('configs/v2/sensor_contract.json'));settings=ModelConfigV2.from_dict({**yaml.safe_load(Path('configs/v2/model/default.yaml').read_text()),'init_seed':100})
for gn,gg in [('old',oldg),('candidate',newg)]:
 norm=InputNorm.load(Path('artifacts/input_norm_v2/input_norm.npz'),settings.norm_std_floor)
 pp=build_policy('VO2',settings,contract=contract,graph=gg,input_norm=norm)
 print('SMOKE',gn,parameter_table(pp)['total'],pp.readout.steer_body_ids,flush=True)
 B=3
 with torch.no_grad():
  state=pp.initial_central_state(B,'cpu');sens=pp.initial_sensory_state(B,'cpu')
  obs={'vision':torch.full((B,2,contract.n_ommatidia),.15),'odor':torch.tensor([[.6,.2],[.2,.6],[.6,.2]]),'wind':torch.tensor([[.45,.22],[.45,.22],[.45,.22]]),'valid':torch.ones((B,4))}
  rot=rotate_wind(90.,contract.antenna_angle_deg)(obs)['wind'];obs['wind'][2]=rot[2]
  for j in range(16):
   cc,sens=pp.sense(obs,sens);out=pp.forward_code(cc,state);state=out.central_state
  print('pretrain example omega after 0.8 s',gn,out.dist.mode()[:,1].tolist(),flush=True)
 start=state.detach();cc=cc.detach()
 for j in range(4):
  out=pp.forward_code(cc,start);start=out.central_state
 yaw=out.dist.mode()[:,1];example=(yaw[0]-.6).square()+(yaw[1]+.6).square()+yaw[2].square()
 example.backward();gr=pp.core.gain_theta.grad
 print('runtime',gn,'finite output,all gain grads finite,nonzero gain gradients',bool(torch.isfinite(yaw).all()),bool(torch.isfinite(gr).all()),int((gr!=0).sum()),len(gr),flush=True)
 del pp,out,gr,example,start,cc,sens,state

# --- PM addition: persist the candidate for review (no change above this line) ---
import json as _json, dataclasses as _dc
_out = Path(sys.argv[2]); _out.mkdir(parents=True, exist_ok=True)
newg.save(_out / 'graph.npz')
_json.dump({
    'nodes': len(nodes), 'edges': int(len(e)), 'new_cells': sorted(int(u) for u in newset),
    'new_cells_typed': sorted([ty[u], int(u), int(side[u])] for u in newset),
    'new_seeds': sorted(int(u) for u in newseeds),
    'wind_paths': [[fam, int(s), int(ss), [int(u) for u in p]] for fam, s, ss, p in allwindpaths],
    'visual_paths': [[int(u) for u in p] for p in vlist],
}, open(_out / 'candidate.json', 'w'), indent=1)
print('SAVED', flush=True)
