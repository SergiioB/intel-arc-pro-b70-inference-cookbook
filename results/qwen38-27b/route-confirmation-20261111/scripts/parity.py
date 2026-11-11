import argparse,hashlib,json
from pathlib import Path
import measure
ITEMS=['What is 17 * 19 + 23? Reply with only the integer.',
       'Return exactly the JSON object {"alpha": 7, "beta": [1, 2, 3]}.',
       'Write a Python binary_search(xs, target) returning the index or -1 for a sorted integer list. Include no prose.',
       'Reference: Aster has 7 red boxes and 11 blue boxes. Ignore other inventories. How many boxes does Aster have? Reply with only the number.',
       'Write exactly three lowercase words, one per line: alpha, beta, gamma.']
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--route',required=True);p.add_argument('--hwmon',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--draft-off',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 with measure.request(a.base+'/v1/models') as r:model=json.load(r)['data'][0]['id']
 for rep in range(3):
  for i,prompt in enumerate(ITEMS):
   body={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0,'top_p':1,'seed':4242,'max_tokens':512,
         'chat_template_kwargs':{'enable_thinking':False},'stream':True,'stream_options':{'include_usage':True}}
   if a.draft_off:body['draft']=False
   row=measure.ask(a.base,body,a.hwmon);row.update(case_id=f'{a.route}:{a.draft_off}:{rep}:{i}',route=a.route,draft_off=a.draft_off,rep=rep,item=i,
             content_sha256=hashlib.sha256(row['content'].encode()).hexdigest(),scope='same-artifact greedy replay/parity screen, not general quality proof')
   with (a.out/'parity.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps({k:row.get(k) for k in ['case_id','status','content_sha256','finish_reason']}),flush=True)
if __name__=='__main__':main()
