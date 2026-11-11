import argparse,concurrent.futures,hashlib,json,time
from pathlib import Path
import measure

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--route',required=True);p.add_argument('--hwmon',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--concurrency',type=int,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 prompt=json.loads((Path(__file__).resolve().parent.parent/'prompts.json').read_text())['code-8k']['content']
 with measure.request(a.base+'/v1/models') as r:model=json.load(r)['data'][0]['id']
 for batch in range(-1,5):
  def one(i):
   nonce=hashlib.sha256(f'C{a.concurrency}:{batch}:{i}'.encode()).hexdigest()
   body={'model':model,'messages':[{'role':'user','content':nonce+'\n'+prompt}],'temperature':.7,'top_p':.8,'top_k':20,'min_p':0,'presence_penalty':0,'seed':4242+i,
         'max_tokens':512,'chat_template_kwargs':{'enable_thinking':False},'stream':True,'stream_options':{'include_usage':True}}
   return measure.ask(a.base,body,a.hwmon)
  e0=measure.read_energy(a.hwmon);t0=time.monotonic()
  with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:rows=list(pool.map(one,range(a.concurrency)))
  wall=time.monotonic()-t0;e1=measure.read_energy(a.hwmon)
  tokens=sum((r.get('usage') or {}).get('completion_tokens',0) for r in rows if r['status']=='ok')
  for row in rows:
   row.pop('energy_j',None);row.pop('average_card_w',None);row['energy_scope']='overlapping request intervals; no per-request energy attribution';row['metric_deltas_scope']='overlapping server counters; not per-request phase timing'
  result={'route':a.route,'concurrency':a.concurrency,'batch':batch,'warmup':batch<0,'wall_s':wall,'aggregate_output_tok_s':tokens/wall,
          'successful_requests':sum(r['status']=='ok' for r in rows),'energy_j':None if e0 is None or e1 is None else (e1-e0)/1e6,'requests':rows}
  with (a.out/f'c{a.concurrency}.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
  print(json.dumps({k:v for k,v in result.items() if k!='requests'}),flush=True)
if __name__=='__main__':main()
