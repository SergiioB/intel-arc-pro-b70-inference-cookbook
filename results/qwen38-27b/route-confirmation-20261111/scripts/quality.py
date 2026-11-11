import argparse, importlib.util, json, resource, subprocess, types
from pathlib import Path
import measure
W=Path(__file__).resolve().parent
SOURCE=Path(__file__).resolve().parents[4]/'benchmarks/quality-suite.py'
spec=importlib.util.spec_from_file_location('mechanical_floor',SOURCE);suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)

def limits():
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
    resource.setrlimit(resource.RLIMIT_CPU,(5,5))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))

def sandbox(argv,**kwargs):
    if argv[1]!='-c':raise RuntimeError('Unsupported grader command')
    return subprocess.run(['bwrap','--unshare-all','--die-with-parent','--ro-bind','/usr','/usr','--ro-bind','/lib','/lib',
       '--ro-bind','/lib64','/lib64','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--chdir','/tmp','/usr/bin/python3','-I','-c',argv[2]],
       preexec_fn=limits,**kwargs)
suite.subprocess=types.SimpleNamespace(run=sandbox,TimeoutExpired=subprocess.TimeoutExpired)

def check(item,answer):
    kind,expected=item[3],item[4]
    if kind in ['exact','needle']:return suite.norm(answer)==suite.norm(expected),'strict full-answer equality'
    if kind=='tool':
        try:
            value=json.loads(answer.strip().removeprefix('```json').removesuffix('```').strip())
            args=value.get('arguments',value.get('args'));args=json.loads(args) if isinstance(args,str) else args
            return value.get('name')==expected[0] and args==expected[1],'strict tool name and JSON arguments'
        except Exception:return False,'not a complete tool-call JSON object'
    return suite.check(item,answer)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);parser.add_argument('--route',required=True)
    parser.add_argument('--hwmon',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mode',choices=['thinking','nonthinking'],default='thinking');parser.add_argument('--reps',type=int,default=3)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    with measure.request(args.base+'/v1/models') as r:model=json.load(r)['data'][0]['id']
    seen=set();path=args.out/'quality.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            try:seen.add(json.loads(line)['case_id'])
            except (ValueError,KeyError):pass
    failures = 0
    for rep in range(args.reps):
        for item in suite.ITEMS:
            case=f'{args.route}:{args.mode}:{rep}:{item[0]}'
            if case in seen:continue
            body={'model':model,'messages':[{'role':'user','content':suite.build_prompt(item)}],
              'temperature':1 if args.mode=='thinking' else .7,'top_p':.95 if args.mode=='thinking' else .8,'top_k':20,
              'min_p':0,'presence_penalty':0,'seed':4242+rep,'max_tokens':8192,'chat_template_kwargs':{'enable_thinking':args.mode=='thinking'},
              'stream':True,'stream_options':{'include_usage':True}}
            row=measure.ask(args.base,body,args.hwmon)
            ok,detail=check(item,row['content']) if row['status']=='ok' else (False,row.get('error'))
            if row['finish_reason']=='length':ok=False;detail='output truncated at common 8192-token budget'
            row.update(case_id=case,area=item[1],item_id=item[0],route=args.route,mode=args.mode,rep=rep,passed=bool(ok),grade_detail=str(detail),
                       grading='strict full answer; code executed without network/home in bwrap; same 40-item floor, not a broad quality ranking')
            with path.open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps({k:row.get(k) for k in ['case_id','status','passed','finish_reason','wall_s','error']}),flush=True)
            failures = failures + 1 if row['status'] != 'ok' else 0
            if failures >= 2: raise RuntimeError('Two consecutive API failures; stopping the quality stage')

    if args.mode == 'nonthinking':
        # Replacement for the nonthinking rep 0 prompts already used in the 256-token smoke.
        subprocess.run([__import__('sys').executable,str(W/'measure.py'),'--base',args.base,'--route',args.route,
            '--out',str(W/'confirmation-extra'/args.route/'nonthinking'),'--hwmon',str(args.hwmon),
            '--prompts',str(W/'prompts.json'),'--depths','0,8k,16k,24k,32k','--reps','1','--rep-start','5',
            '--mode','nonthinking','--max-tokens','512'],check=True)
        subprocess.run([__import__('sys').executable,str(W/'broad.py'),'--base',args.base,'--route',args.route,'--out',str(args.out/'broad84'),'--hwmon',str(args.hwmon)],check=True)

if __name__=='__main__':main()
