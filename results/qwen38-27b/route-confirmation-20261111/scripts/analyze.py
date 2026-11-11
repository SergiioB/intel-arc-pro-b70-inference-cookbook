#!/usr/bin/env python3
"""Condense retained evidence after the run. Does not rewrite any measurement."""
import argparse,csv,json,random,statistics
from datetime import datetime,timezone
from collections import defaultdict
from pathlib import Path

def rows(path):
 if not path.exists():return
 with path.open() as f:
  for line in f:
   try:yield json.loads(line)
   except ValueError:pass

def median(values):
 values=[v for v in values if v is not None]
 return statistics.median(values) if values else None

def percentile(values,p):
 values=sorted(v for v in values if v is not None)
 if not values:return None
 at=(len(values)-1)*p;lo=int(at);hi=min(lo+1,len(values)-1)
 return values[lo]+(values[hi]-values[lo])*(at-lo)

def bootstrap(values,seed=4242):
 if not values:return None
 rng=random.Random(seed);n=len(values)
 boot=sorted(sum(rng.choices(values,k=n))/n for _ in range(3000))
 return [percentile(boot,.025),percentile(boot,.975)]

def broad_summary(root):
 # Keep API incompatibility and grader failures visible alongside paired correctness.
 unique={}
 for path in (root/'quality').rglob('broad.jsonl') if (root/'quality').exists() else []:
  for row in rows(path):unique[row['case_id']]=row
 paired=defaultdict(dict);areas=defaultdict(lambda:defaultdict(list));health=defaultdict(lambda:defaultdict(int))
 for row in unique.values():
  route=row['route'];areas[row['area']][route].append(row)
  h=health[route];h['attempted']+=1
  h['api_errors']+=bool(row.get('api_error'));h['http_400']+=row.get('http_code')==400
  h['grader_errors']+=bool(row.get('grader_error'));h['truncated']+=row.get('finish_reason')=='length'
  h['passed']+=row['grade'].get('passed') is True
  paired[(row['item_id'],row['rep'])][route]=row
 comparisons=[]
 for area in ['all']+sorted(areas):
  differences=defaultdict(list);flips=defaultdict(int);n=0;excluded=defaultdict(int)
  for (item,rep),data in paired.items():
   if not all(route in data for route in ['tf','exl3']):continue
   if area!='all' and data['tf']['area']!=area:continue
   if any(r.get('api_error') for r in data.values()):excluded['api_error_pairs']+=1;continue
   if any(r.get('grader_error') or r['grade'].get('passed') is None for r in data.values()):excluded['grader_error_pairs']+=1;continue
   tf=data['tf']['grade']['passed'] is True;ex=data['exl3']['grade']['passed'] is True;n+=1
   flips['both_pass' if tf and ex else 'both_fail' if not tf and not ex else 'tf_only' if tf else 'exl3_only']+=1
   differences[item].append(int(ex)-int(tf))
  comparisons.append({'area':area,'paired_outcomes':n,'distinct_items':len(differences),'paired_flips':dict(flips),'excluded':dict(excluded),
    'item_bootstrap_exl3_minus_tf_pass_fraction_pointwise_95_interval':bootstrap([statistics.mean(x) for x in differences.values()])})
 categories=[]
 for area,data in sorted(areas.items()):
  for route,values in sorted(data.items()):
   graded=[r for r in values if not r.get('api_error') and not r.get('grader_error') and r['grade'].get('passed') is not None]
   successful=[r for r in graded if r['grade']['passed']]
   categories.append({'area':area,'route':route,'attempted':len(values),'mechanically_evaluable':len(graded),'passed':len(successful),
     'api_errors':sum(bool(r.get('api_error')) for r in values),'grader_errors':sum(bool(r.get('grader_error')) for r in values),
     'truncated':sum(r.get('finish_reason')=='length' for r in values),
     'median_success_wall_s':median(r.get('wall_s') for r in successful),
     'energy_per_success_j':sum(r.get('energy_j') or 0 for r in graded)/len(successful) if successful and all(r.get('energy_j') is not None for r in graded) else None})
 return {'scope':'Frozen 84-item audited subset, nonthinking, three repeats; API compatibility separate. No universal quality claim.',
   'health':dict(health),'categories':categories,'paired_comparisons':comparisons}

def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
 groups=defaultdict(list);paired=defaultdict(dict);flat=[];failures=[];seen_cells=set();exploratory_hashes=set();excluded_exploratory=[]
 for folder in ['smoke-tf','smoke-exl3']:
  for row in rows(root/folder/'requests.jsonl'):
   if row.get('prompt_sha256'):exploratory_hashes.add(row['prompt_sha256'])
 for block in ['A1','B1','B2','A2','confirmation-extra','prose','capacity-240k']:
  for path in (root/block).rglob('requests.jsonl') if (root/block).exists() else []:
   for row in rows(path):
    if row.get('warmup'):continue
    if row.get('prompt_sha256') in exploratory_hashes:
     excluded_exploratory.append(row['cell_id']);continue
    if row['cell_id'] in seen_cells:raise RuntimeError('Duplicate measured cell: '+row['cell_id'])
    seen_cells.add(row['cell_id'])
    use=row.get('status')=='ok' and row.get('usage_valid') and (row['phase']!='cold' or row.get('cold_valid'))
    item={k:row.get(k) for k in ['route','mode','kind','depth','rep','phase','status','ttft_s','first_answer_s','client_post_first_estimate_tok_s','e2e_output_tok_s','energy_j','cached_tokens','cache_hit_delta','cold_valid','temp_start_mc','temp_end_mc','cap_uw','finish_reason','prompt_sha256']}
    item.update(prompt_tokens=(row.get('usage') or {}).get('prompt_tokens'),completion_tokens=(row.get('usage') or {}).get('completion_tokens'),valid_for_comparison=bool(use),block=block)
    flat.append(item)
    if not use:failures.append({'cell_id':row.get('cell_id'),'error':row.get('error'),'cold_valid':row.get('cold_valid')});continue
    key=(row['route'],row['mode'],row['kind'],row['depth'],row['phase']);groups[key].append(item)
    paired[(row['mode'],row['kind'],row['depth'],row['phase'],row['rep'])][row['route']]=item
 summary=[]
 for key,data in sorted(groups.items()):
  summary.append(dict(zip(['route','mode','kind','depth','phase'],key))|{'n':len(data),'reps':sorted(r['rep'] for r in data),
   'median_ttft_s':median(r['ttft_s'] for r in data),'median_first_answer_s':median(r['first_answer_s'] for r in data),
   'answers_within_speed_budget':sum(r['first_answer_s'] is not None for r in data),
   'median_client_post_first_estimate_tok_s':median(r['client_post_first_estimate_tok_s'] for r in data),
   'median_e2e_output_tok_s':median(r['e2e_output_tok_s'] for r in data),
   'median_energy_j':median(r['energy_j'] for r in data),
   'truncated_at_speed_budget':sum(r.get('finish_reason')=='length' for r in data),
   'prompt_tokens_range':[min(r['prompt_tokens'] for r in data),max(r['prompt_tokens'] for r in data)],
   'temperature_start_range_c':[min(r['temp_start_mc'] for r in data)/1000,max(r['temp_start_mc'] for r in data)/1000]})
 ratios=defaultdict(list);timing_deltas=defaultdict(list);token_mismatches=[]
 for key,data in paired.items():
  if not all(r in data for r in ['tf','exl3']):continue
  tf,ex=data['tf'],data['exl3'];group=key[:4]
  if tf['prompt_sha256']!=ex['prompt_sha256']:raise RuntimeError('Unmatched prompt bytes: '+str(key))
  if tf['prompt_tokens']!=ex['prompt_tokens']:token_mismatches.append({'case':key,'tf':tf['prompt_tokens'],'exl3':ex['prompt_tokens']})
  if tf['client_post_first_estimate_tok_s'] and ex['client_post_first_estimate_tok_s']:
   ratios[group].append(ex['client_post_first_estimate_tok_s']/tf['client_post_first_estimate_tok_s'])
  if tf['ttft_s'] is not None and ex['ttft_s'] is not None:timing_deltas[group].append(ex['ttft_s']-tf['ttft_s'])
 comparisons=[]
 for key,values in sorted(ratios.items()):
  ci=bootstrap(values)
  verdict='inconclusive'
  if len(values)>=5 and ci[0]>1.1:verdict='EXL3 clears the 10% margin in this shape (pointwise interval, generation-rate estimate)'
  elif len(values)>=5 and ci[1]<1/1.1:verdict='TensorFold clears the 10% margin in this shape (pointwise interval, generation-rate estimate)'
  elif len(values)>=5:verdict='no resolved 10% generation-rate difference'
  comparisons.append(dict(zip(['mode','kind','depth','phase'],key))|{'paired_n':len(values),'ratio_exl3_over_tf_mean':statistics.mean(values),'paired_repeat_bootstrap_pointwise_95_interval':ci,
    'median_ttft_delta_exl3_minus_tf_s':median(timing_deltas[key]),'verdict':verdict})
 quality=[];qrows=defaultdict(dict)
 for path in (root/'quality').rglob('quality.jsonl') if (root/'quality').exists() else []:
  for r in rows(path):qrows[(r['mode'],r['rep'],r['item_id'])][r['route']]=r
 for mode in ['thinking','nonthinking']:
  peritem=defaultdict(list);flips={'tf_only':0,'exl3_only':0,'both_pass':0,'both_fail':0};totals={'tf':0,'exl3':0};trunc={'tf':0,'exl3':0};n=0
  api_failures={'tf':0,'exl3':0}
  for (m,rep,item),data in qrows.items():
   if m!=mode or not all(k in data for k in ['tf','exl3']):continue
   if any(r.get('status')!='ok' for r in data.values()):
    for route in api_failures:api_failures[route]+=data[route].get('status')!='ok'
    continue
   n+=1;tf=bool(data['tf']['passed']);ex=bool(data['exl3']['passed']);totals['tf']+=tf;totals['exl3']+=ex
   for route in totals:trunc[route]+=data[route].get('finish_reason')=='length'
   flips['both_pass' if tf and ex else 'both_fail' if not tf and not ex else 'tf_only' if tf else 'exl3_only']+=1
   peritem[item].append(int(ex)-int(tf))
  if n:
   differences=[statistics.mean(x) for x in peritem.values()]
   quality.append({'mode':mode,'paired_outcomes':n,'distinct_items':len(peritem),'passes':totals,'api_failures_in_excluded_pairs':api_failures,'paired_flips':flips,'truncated':trunc,
     'item_bootstrap_exl3_minus_tf_pass_fraction_95_interval':bootstrap(differences),
     'scope':'40-item floor only; repeats within items are correlated; not a universal quality ranking'})
 sessions=[]
 for path in (root/'sessions').rglob('turns.jsonl') if (root/'sessions').exists() else []:
  for r in rows(path):sessions.append({k:r.get(k) for k in ['route','depth','rep','turn','ttft_s','first_answer_s','passed','cached_tokens','status']})
 retrieval=[]
 for path in (root/'retrieval').rglob('retrieval.jsonl') if (root/'retrieval').exists() else []:
  for r in rows(path):retrieval.append({k:r.get(k) for k in ['route','depth','rep','passed','correct_keys','ttft_s','status']})
 parity=defaultdict(list)
 for path in (root/'parity').rglob('parity.jsonl') if (root/'parity').exists() else []:
  for r in rows(path):parity[(r['route'],r['draft_off'],r['item'])].append(r['content_sha256'])
 hard=[]
 for path in (root/'hard-code').rglob('hard-code.jsonl') if (root/'hard-code').exists() else []:
  for r in rows(path):hard.append({k:r.get(k) for k in ['route','item_id','rep','passed','wall_s','finish_reason','status']})
 concurrency=[]
 for path in (root/'concurrency').rglob('c*.jsonl') if (root/'concurrency').exists() else []:
  batches=[r for r in rows(path) if not r.get('warmup')]
  if not batches:continue
  first=batches[0];reqs=[req for batch in batches for req in batch['requests']]
  concurrency.append({'route':first['route'],'concurrency':first['concurrency'],'batches':len(batches),
     'median_aggregate_output_tok_s':median(r['aggregate_output_tok_s'] for r in batches),
     'requests':len(reqs),'failed':sum(r['status']!='ok' for r in reqs),
     'ttft_p50_s':percentile([r['ttft_s'] for r in reqs],.5),'ttft_p95_s':percentile([r['ttft_s'] for r in reqs],.95),
     'ttft_p99_s':percentile([r['ttft_s'] for r in reqs],.99),'scope':'small n; percentile estimates are descriptive, not a tail-latency SLO'})
 output={'generated_utc':datetime.now(timezone.utc).isoformat(),'inference_limits':'Pointwise bootstrap intervals are descriptive, not multiplicity adjusted. Five repeats characterize these frozen prompts and deployed stacks, not all workloads. Generation-rate estimates include reasoning where enabled.', 'complete_marker':(root/'COMPLETE.json').exists(),'conditions':'one B70, 180 W, C1 except labelled concurrency; deployed quantized/speculative stacks',
   'speed_cells':summary,'paired_comparisons':comparisons,'excluded_exploratory_prompt_cells':excluded_exploratory,'invalid_cells':failures,'prompt_token_mismatches':token_mismatches,
   'broad_quality':broad_summary(root),'quality':quality,'sessions':sessions,'retrieval':retrieval,'hard_code':hard,'concurrency':concurrency,
   'parity':[{'route':key[0],'draft_off':key[1],'item':key[2],'n':len(hashes),'distinct_output_hashes':len(set(hashes)),'hashes':hashes} for key,hashes in parity.items()]}
 (root/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
 if flat:
  with (root/'speed-cells.csv').open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
 print(json.dumps({'complete':output['complete_marker'],'valid_speed_groups':len(summary),'paired_groups':len(comparisons),'invalid_cells':len(failures),'quality_modes':len(quality),'session_turns':len(sessions)}))
if __name__=='__main__':main()
