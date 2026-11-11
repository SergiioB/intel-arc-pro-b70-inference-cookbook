import argparse,importlib.util,json
from pathlib import Path
import measure
from quality import sandbox

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--route',required=True);p.add_argument('--hwmon',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 spec=importlib.util.spec_from_file_location('hard',str(Path(__file__).with_name('creative_suite.py')));suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)
 with measure.request(a.base+'/v1/models') as r:model=json.load(r)['data'][0]['id']
 dest=a.out/'hard-code.jsonl'
 seen={json.loads(line)['case_id'] for line in dest.read_text().splitlines()} if dest.exists() else set()
 for rep in range(3):
  for item in suite.ITEMS:
   if item[0] not in ['py-xml','py-diff','py-jsonpath']:continue
   name,kind,prompt=item
   if f'{a.route}:{name}:{rep}' in seen:continue
   body={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':.7,'top_p':.8,'top_k':20,'min_p':0,'presence_penalty':0,'seed':4242+rep,
         'max_tokens':8192,'chat_template_kwargs':{'enable_thinking':False},'stream':True,'stream_options':{'include_usage':True}}
   row=measure.ask(a.base,body,a.hwmon);code=suite.extract(row['content'],kind)
   try:
    result=sandbox(['python','-c',(code or '')+'\n'+suite.TESTS[name]],capture_output=True,text=True,timeout=20)
    passed=bool(code) and result.returncode==0 and 'ok' in result.stdout
    detail={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
   except Exception as e:passed=False;detail={'error':str(e)}
   if row['finish_reason']=='length':passed=False
   row.update(case_id=f'{a.route}:{name}:{rep}',route=a.route,item_id=name,rep=rep,passed=passed,grade=detail,scope='three harder Python tasks against pre-existing tests; nonthinking, 8192-token common cap')
   with (a.out/'hard-code.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps({k:row.get(k) for k in ['case_id','passed','status','wall_s','finish_reason']}),flush=True)
if __name__=='__main__':main()
