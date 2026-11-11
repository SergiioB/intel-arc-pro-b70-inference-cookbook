import argparse,hashlib,json
from pathlib import Path
from tokenizers import Tokenizer
import measure

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--route',required=True);p.add_argument('--hwmon',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True,help='tokenizer.json for the EXL3 artifact used to build the prompts');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 tok=Tokenizer.from_file(str(a.tokenizer))
 source=json.loads((Path(__file__).resolve().parent.parent/'prompts.json').read_text())['code-240k']['content'];ids=tok.encode(source).ids
 with measure.request(a.base+'/v1/models') as r:model=json.load(r)['data'][0]['id']
 for depth,n in [('32k',32768),('120k',122880),('near-limit',130400)]+([('240k',245760)] if a.route=='exl3' else []):
  for rep in range(3):
   seed=hashlib.sha256(f'needle-{depth}-{rep}'.encode()).hexdigest();expected={k:seed[i*8:i*8+8] for i,k in enumerate(['alpha','beta','gamma'])}
   sections=[];last=0
   for fraction,key in zip([.1,.5,.9],expected):
    at=int((n-400)*fraction);sections.append(tok.decode(ids[last:at]));sections.append(f'\nAUDIT RECORD {key}: {expected[key]}\n');last=at
   sections.append(tok.decode(ids[last:n-400]));text='Trial '+seed+'\n'+''.join(sections)+'\nReturn only a JSON object mapping alpha, beta and gamma to their AUDIT RECORD values.'
   body={'model':model,'messages':[{'role':'user','content':text}],'temperature':.7,'top_p':.8,'top_k':20,'min_p':0,'presence_penalty':0,'seed':4242+rep,
         'max_tokens':128,'chat_template_kwargs':{'enable_thinking':False},'stream':True,'stream_options':{'include_usage':True}}
   row=measure.ask(a.base,body,a.hwmon)
   try:answer=json.loads(row['content'].strip())
   except ValueError:answer={}
   row.update(case_id=f'{a.route}:{depth}:{rep}',route=a.route,depth=depth,rep=rep,expected=expected,correct_keys=sum(answer.get(k)==v for k,v in expected.items()),passed=answer==expected,scope='three planted codes near 10/50/90 percent of tokenized reference, same prompts across stacks, no drafts-off quality inference')
   with (a.out/'retrieval.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps({k:row.get(k) for k in ['case_id','passed','correct_keys','status','usage','error']}),flush=True)
if __name__=='__main__':main()
