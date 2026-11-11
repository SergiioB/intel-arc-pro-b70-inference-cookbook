#!/usr/bin/env python3
"""Strict streamed comparison client. Records failures; never counts SSE events as tokens."""
import argparse, hashlib, json, time, urllib.error, urllib.request
from pathlib import Path

def request(url, payload=None, timeout=1200):
    req=urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
    return urllib.request.urlopen(req,timeout=timeout)

def read_energy(hw):
    try:return int((hw/'energy1_input').read_text())
    except OSError:return None

def snapshot(base):
    try:
        with request(base+'/metrics',timeout=10) as r: raw=r.read().decode()
        values={}
        for line in raw.splitlines():
            if not line or line.startswith('#'):continue
            name, value=line.rsplit(None,1)
            if any(term in name for term in ['prefix_cache_hits','request_decode_time_seconds_sum','request_prefill_time_seconds_sum']):
                key=name.split('{')[0];values[key]=values.get(key,0)+float(value)
        return {'counters':values,'raw':raw}
    except Exception as e:return {'error':str(e),'counters':{}}

def ask(base, payload, hw):
    before=snapshot(base); e0=read_energy(hw); start=time.monotonic()
    row={'utc':time.strftime('%FT%TZ',time.gmtime()),'request':payload,'metrics_before':before}
    first=last=first_content=first_reasoning=None
    content=[];reasoning=[]; events=[];usage=None;finish=None;native=None
    try:
        with request(base+'/v1/chat/completions',payload) as response:
            for line in response:
                line=line.decode().strip()
                if not line.startswith('data:'):continue
                data=line[5:].strip()
                if data=='[DONE]':break
                item=json.loads(data);at=time.monotonic()
                events.append({'at_s':at-start,'data':item})
                if item.get('error'):raise RuntimeError(str(item['error']))
                if item.get('usage'):usage=item['usage']
                if item.get('tensorfold'):native=item['tensorfold']
                for choice in item.get('choices') or []:
                    delta=choice.get('delta') or {}
                    c=delta.get('content') or '';r=delta.get('reasoning_content') or delta.get('reasoning') or ''
                    if c or r:
                        first=at if first is None else first;last=at
                    if c:
                        first_content=at if first_content is None else first_content;content.append(c)
                    if r:
                        first_reasoning=at if first_reasoning is None else first_reasoning;reasoning.append(r)
                    finish=choice.get('finish_reason') or finish
        row['status']='ok'
    except Exception as error:
        row['status']='error'; row['error']=str(error)
        if isinstance(error,urllib.error.HTTPError):row['error_body']=error.read().decode(errors='replace')
    end=time.monotonic();e1=read_energy(hw);after=snapshot(base)
    row.update(wall_s=end-start,usage=usage,finish_reason=finish,
               ttft_s=None if first is None else first-start,
               first_answer_s=None if first_content is None else first_content-start,
               first_reasoning_s=None if first_reasoning is None else first_reasoning-start,
               content=''.join(content),reasoning=''.join(reasoning),sse=events,native_tensorfold=native,metrics_after=after)
    row['output_sha256']=hashlib.sha256((row['reasoning']+'\n'+row['content']).encode()).hexdigest()
    if e0 is not None and e1 is not None and e1>=e0:
        row['energy_j']=(e1-e0)/1e6; row['average_card_w']=row['energy_j']/(end-start)
    row['metric_deltas']={key:value-before['counters'][key] for key,value in after['counters'].items() if key in before['counters']}
    n=(usage or {}).get('completion_tokens')
    row['usage_valid']=isinstance(n,int) and n>0 and isinstance((usage or {}).get('prompt_tokens'),int)
    if row['usage_valid']:
        row['e2e_output_tok_s']=n/(end-start)
        # First SSE burst may contain >1 token. Explicitly label this estimate.
        if first is not None and last>first:row['client_post_first_estimate_tok_s']=(n-1)/(last-first)
    if row['status']=='ok' and (not row['usage_valid'] or first is None):
        row['status']='invalid';row['error']='Missing output, actual token counts, or first-token timestamp'
    row['cached_tokens']=((usage or {}).get('prompt_tokens_details') or {}).get('cached_tokens')
    row['cache_hit_delta']=sum(v for k,v in row['metric_deltas'].items() if 'prefix_cache_hits' in k) if any('prefix_cache_hits' in k for k in row['metric_deltas']) else None
    return row

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);parser.add_argument('--route',required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--hwmon',type=Path,required=True)
    parser.add_argument('--depths',default='0,8k,16k,24k,32k,64k,120k');parser.add_argument('--reps',type=int,default=5);parser.add_argument('--rep-start',type=int,default=0)
    parser.add_argument('--mode',choices=['thinking','nonthinking'],default='thinking');parser.add_argument('--max-tokens',type=int,default=512)
    parser.add_argument('--kinds',default='code');parser.add_argument('--phase',choices=['cold','paired-warm'],default='paired-warm')
    parser.add_argument('--prompts',type=Path,required=True);parser.add_argument('--draft-off',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    with request(args.base+'/v1/models') as r:model=json.load(r)['data'][0]['id']
    prompts=json.loads(args.prompts.read_text()); seen=set()
    dest=args.out/'requests.jsonl'
    if dest.exists():
        for line in dest.read_text().splitlines():
            try:seen.add(json.loads(line)['cell_id'])
            except (ValueError,KeyError):pass
    mode=args.mode=='thinking'
    base={'model':model,'temperature':1.0 if mode else .7,'top_p':.95 if mode else .8,'top_k':20,'min_p':0,'presence_penalty':0,
          'max_tokens':args.max_tokens,'chat_template_kwargs':{'enable_thinking':mode},'stream':True,'stream_options':{'include_usage':True}}
    if args.draft_off:base['draft']=False
    for depth in args.depths.split(','):
        for kind in args.kinds.split(','):
            for rep in [-1-args.rep_start]+list(range(args.rep_start,args.rep_start+args.reps)):
                item=prompts[f'{kind}-{depth}'];seed=4242+max(rep,0)
                nonce=hashlib.sha256(f'confirmation:{args.mode}:{kind}:{depth}:{rep}'.encode()).hexdigest()
                prompt='Request identifier (not part of the task): '+nonce+'\n'+item['content']
                payload={**base,'seed':seed,'messages':[{'role':'user','content':prompt}]}
                for phase in ['cold','warm'] if args.phase=='paired-warm' else ['cold']:
                    cell=f'{args.route}:{args.mode}:{kind}:{depth}:{rep}:{phase}:{args.draft_off}'
                    if cell in seen:continue
                    temp_start=int((args.hwmon/'temp2_input').read_text()); row=ask(args.base,payload,args.hwmon); row["temp_start_mc"]=temp_start
                    row.update(cell_id=cell,route=args.route,mode=args.mode,kind=kind,depth=depth,rep=rep,phase=phase,warmup=rep<0,
                               prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),temp_end_mc=int((args.hwmon/'temp2_input').read_text()),
                               cap_uw=int((args.hwmon/'power1_cap').read_text()),thermal_regime='sustained; per-cell temperatures retained')
                    if phase=='cold' and (row['cached_tokens'] not in [None,0] or row['cache_hit_delta'] not in [None,0]):row['cold_valid']=False
                    else:row['cold_valid']=phase=='cold' and (row['cached_tokens']==0 or row['cache_hit_delta']==0)
                    with dest.open('a') as stream:stream.write(json.dumps(row)+'\n')
                    print(json.dumps({k:row.get(k) for k in ['cell_id','status','usage','ttft_s','first_answer_s','client_post_first_estimate_tok_s','cached_tokens','cold_valid','error']}),flush=True)
                    if row['status']!='ok':raise RuntimeError(f"Invalid measured cell {cell}: {row.get('error')}")

if __name__=='__main__':main()
