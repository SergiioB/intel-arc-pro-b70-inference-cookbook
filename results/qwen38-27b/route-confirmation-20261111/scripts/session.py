import argparse,hashlib,json
from pathlib import Path
import measure

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--route',required=True);p.add_argument('--hwmon',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 prompts=json.loads((Path(__file__).resolve().parent.parent/'prompts.json').read_text())
 with measure.request(a.base+'/v1/models') as r:model=json.load(r)['data'][0]['id']
 dest=a.out/'turns.jsonl';existing={}
 if dest.exists():existing={json.loads(l)['case_id']:json.loads(l) for l in dest.read_text().splitlines()}
 for depth in ['8k','64k']:
  for rep in range(3):
   seed=hashlib.sha256(f'session-{depth}-{rep}'.encode()).hexdigest()
   expected={'alpha':seed[:8],'beta':seed[8:16],'gamma':seed[16:24]}
   source=prompts['chat-'+depth]['content'];size=len(source)
   # Preserve the document across turns; plant exact-answer facts throughout it.
   for fraction,key in reversed(list(zip([.1,.5,.9],expected))):
    at=int(size*fraction);source=source[:at]+f'\nAUDIT RECORD {key}: {expected[key]}\n'+source[at:]
   messages=[{'role':'user','content':'Session '+seed+'\nRead this reference document:\n'+source+'\nReturn only the JSON object mapping alpha, beta and gamma to their AUDIT RECORD values.'}]
   for turn in range(10):
    case=f'{a.route}:{depth}:{rep}:{turn}'
    payload={'model':model,'messages':messages,'temperature':.7,'top_p':.8,'top_k':20,'min_p':0,'presence_penalty':0,'seed':4242+rep,
             'max_tokens':256,'chat_template_kwargs':{'enable_thinking':False},'stream':True,'stream_options':{'include_usage':True}}
    if case in existing:
     messages += [{'role':'assistant','content':existing[case]['content']},{'role':'user','content':f'Follow-up {turn+1}: Return only the same three original AUDIT RECORD values as a JSON object. Use the reference, not any previous answer.'}]
     continue
    row=measure.ask(a.base,payload,a.hwmon)
    try:correct=json.loads(row['content'].strip())==expected
    except ValueError:correct=False
    row.update(case_id=case,route=a.route,depth=depth,rep=rep,turn=turn,expected=expected,passed=correct,scope='same 8K/64K reference held in growing message history; nonthinking; three sessions')
    with dest.open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps({k:row.get(k) for k in ['case_id','passed','ttft_s','cached_tokens','status']}),flush=True)
    if row['status']!='ok':break
    messages += [{'role':'assistant','content':row['content']},{'role':'user','content':f'Follow-up {turn+1}: Return only the same three original AUDIT RECORD values as a JSON object. Use the reference, not any previous answer.'}]
if __name__=='__main__':main()
