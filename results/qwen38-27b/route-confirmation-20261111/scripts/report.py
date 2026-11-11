#!/usr/bin/env python3
"""Render retained measurements; visibly distinguish incomplete from confirmed cells."""
import argparse,html,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'tf':'#eaa364','exl3':'#63cce3'}
NAMES={'tf':'TensorFold + DFlash2','exl3':'EXL3 + MTP3'}
DEPTHS=['0','8k','16k','24k','32k','64k','120k']

def table(headers,rows):
 return '| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)

def fmt(value,precision=2):return '—' if value is None else f'{value:.{precision}f}'

def save(fig,root,name):
 fig.savefig(root/(name+'.png'),dpi=170,bbox_inches='tight');fig.savefig(root/(name+'.svg'),bbox_inches='tight');plt.close(fig)

def failure_review(root):
 review=json.loads((root/'quality-failures.json').read_text())
 rows=[[r['case_id'],r['status'],r['finish_reason'],(r.get('usage') or {}).get('completion_tokens'),r['visible_content_chars'],r['visible_reasoning_chars'],r['sse_events']] for r in review['records']]
 lines=['The two excluded EXL3 nonthinking `tool-04` repeats returned HTTP 200 and usage counts but no visible content. They are excluded from paired scoring, not scored as model failures, and do not establish a route quality difference.',table(['Case','Status','Finish','Completion tokens','Visible content chars','Visible reasoning chars','SSE events'],rows)]
 control=review.get('control')
 if control:
  lines.append(f"Control: `{control['case_id']}` with the same prompt returned the requested JSON tool call ({control['usage']['completion_tokens']} completion tokens, {control['visible_content_chars']} visible characters). The retained SSE for the failing repeats contains only empty deltas and a `stop`; no `tool_calls` or `reasoning_content` field was sent. This is consistent with server-side suppression of special/tool-call tokens, but raw server token ids were not retained, so that cause is not confirmed. It is not an HTTP or transport error.")
 return '\n\n'.join(lines)

def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
 data=json.loads((root/'summary.json').read_text());complete=data['complete_marker']
 plt.rcParams.update({'figure.facecolor':'#11151b','axes.facecolor':'#11151b','text.color':'#f2f4f7','axes.labelcolor':'#b8c4d2','xtick.color':'#b8c4d2','ytick.color':'#b8c4d2','axes.edgecolor':'#465364','axes.titlecolor':'#f2f4f7','savefig.facecolor':'#11151b','font.size':10})
 status='CAMPAIGN FINISHED — inspect failures below' if complete else 'INCOMPLETE — no final route recommendation'
 charts=[]
 for mode in ['thinking','nonthinking']:
  cells=[r for r in data['speed_cells'] if r['mode']==mode and r['kind']=='code']
  if not cells:continue
  fig,axes=plt.subplots(2,2,figsize=(13,8.5));fig.subplots_adjust(hspace=.44,wspace=.25,top=.82,bottom=.14)
  fig.suptitle(f'Qwen3.8-27B: context scaling ({mode})',fontsize=19,y=.98)
  fig.text(.5,.91,status,ha='center',color='#eaa364',fontsize=10)
  for col,phase in enumerate(['cold','warm']):
   for row,metric,label in [(0,'median_client_post_first_estimate_tok_s','Generation rate estimate (tokens/s)'),(1,'median_ttft_s','Time to first generated text (seconds)')]:
    ax=axes[row,col]
    for route in ['tf','exl3']:
     selected={r['depth']:r for r in cells if r['phase']==phase and r['route']==route}
     points=[(i,selected[d]) for i,d in enumerate(DEPTHS) if d in selected and selected[d][metric] is not None]
     if not points:continue
     ax.plot([i for i,r in points],[r[metric] for i,r in points],color=COLORS[route],marker='o',label=NAMES[route])
     for i,r in points:ax.annotate('n='+str(r['n']),(i,r[metric]),xytext=(0,8 if route=='tf' else -14),textcoords='offset points',ha='center',fontsize=8,color=COLORS[route])
    ax.set_title('Unique prefix / cold' if phase=='cold' else 'Immediate identical-prompt replay')
    ax.set_xticks(range(len(DEPTHS)),['Short']+DEPTHS[1:]);ax.set_xlabel('Input target (actual token counts retained)');ax.set_ylabel(label);ax.set_ylim(bottom=0);ax.grid(alpha=.15)
    if ax.lines:ax.legend(fontsize=8)
  note='One headless B70 · 180 W · C1 · deployed quantized stacks · 512 generated tokens\nThinking tokens count as generation. First-text latency is not time to a final answer.\nSpeculative SSE bursts make client generation rates estimates. Warm replay can still miss cache.'
  fig.text(.5,.035,note,ha='center',color='#b8c4d2',fontsize=9)
  name='context-'+mode;save(fig,root,name);charts.append(name)
 # Multi-turn latency uses one line per independent session, retaining instability.
 turns=data['sessions']
 if turns:
  fig,axes=plt.subplots(1,2,figsize=(12,5));fig.subplots_adjust(bottom=.24,top=.78)
  fig.suptitle('Follow-up latency over a 10-turn document chat',fontsize=17)
  for ax,depth in zip(axes,['8k','64k']):
   for route in ['tf','exl3']:
    for rep in sorted({r['rep'] for r in turns if r['route']==route and r['depth']==depth}):
     rs=sorted([r for r in turns if r['route']==route and r['depth']==depth and r['rep']==rep and r.get('first_answer_s') is not None],key=lambda r:r['turn'])
     ax.plot([r['turn'] for r in rs],[r['first_answer_s'] for r in rs],color=COLORS[route],marker='.',alpha=.7,label=NAMES[route] if rep==0 else None)
   ax.set_title(depth+' document');ax.set_xlabel('Turn');ax.set_ylabel('Time to first answer text (seconds)');ax.set_ylim(bottom=0);ax.grid(alpha=.15)
   if ax.lines:ax.legend(fontsize=8)
  fig.text(.5,.055,status+'\nNonthinking · each line = one session · answer correctness reported separately',ha='center',color='#b8c4d2',fontsize=9)
  save(fig,root,'chat-over-time');charts.append('chat-over-time')
 # Category denominators are explicit; missing routes have no bar, never a zero score.
 broad=data['broad_quality'];categories=broad['categories']
 if categories:
  areas=sorted({r['area'] for r in categories});fig,ax=plt.subplots(figsize=(13,7));fig.subplots_adjust(left=.2,bottom=.23,top=.84)
  for route,offset in [('tf',-.18),('exl3',.18)]:
   lookup={r['area']:r for r in categories if r['route']==route}
   for i,area in enumerate(areas):
    if area not in lookup:continue
    r=lookup[area];n=r['mechanically_evaluable']
    if not n:continue
    score=100*r['passed']/n;ax.barh(i+offset,score,height=.32,color=COLORS[route],label=NAMES[route] if i==0 else None)
    ax.text(min(score+1,94),i+offset,f"{r['passed']}/{n}",va='center',fontsize=8)
  ax.set_yticks(range(len(areas)),[x.replace('_',' ') for x in areas]);ax.set_xlim(0,110);ax.set_xlabel('Mechanically graded pass rate (%)');ax.legend(loc='lower right',fontsize=9)
  fig.suptitle('Frozen broad quality subset — nonthinking',fontsize=18)
  fig.text(.5,.07,status+'\n84 distinct items × 3 repeats per route. Repeats are correlated. API/grader failures excluded here and tabulated below.\nDifferent quantizations and speculative stacks: this does not isolate runtime or model weights.',ha='center',fontsize=9,color='#b8c4d2')
  save(fig,root,'quality-by-area');charts.append('quality-by-area')
 text=['# B70 EXL3 / TensorFold confirmation',f'**{status}.** Generated {data["generated_utc"]}.','This compares the exact deployed Qwen3.8-27B stacks on one headless B70 at 180 W. It does not establish a universal best model or isolate runtime from quantization and speculative decoding.',
  '## Context and serving limits',table(['Stack','Configured total context','Meaning'],[['EXL3 4 bpw + MTP3','262,144 tokens','Input + output; acceptance and retrieval checks below'],['TensorFold 4-bit g64 + DFlash2','131,072 tokens','Input + output; 2 GiB prefix cache']]),
  'These are the tested serving configurations, not theoretical maximum contexts of every possible build. Long-input retrieval checks test planted facts; they do not prove all long-context reasoning is reliable.',
  '## Speed evidence',table(['Mode','Corpus','Input target','Replay','Pairs','EXL3 / TF generation rate','Assessment'],[[r['mode'],r['kind'],r['depth'],r['phase'],r['paired_n'],fmt(r['ratio_exl3_over_tf_mean']),r['verdict']] for r in data['paired_comparisons']]),
  data['inference_limits'],
  '## Actual answer delivery',table(['Route','Mode','Input','Replay','n','First answer observed','Budget exhausted'],[[r['route'],r['mode'],r['depth'],r['phase'],r['n'],str(r['answers_within_speed_budget'])+'/'+str(r['n']),r['truncated_at_speed_budget']] for r in data['speed_cells']]),
  'The 512-token speed probes intentionally cap generation. A missing final answer in thinking mode is recorded as missing, never as zero latency. Quality tasks use a separate 8,192-token cap and record truncations as failures.',
  '## Multi-turn sessions',
  table(['Route','Document','Sessions','Turns','Correct after re-grade'],[ [route,depth,len({r['rep'] for r in data['sessions'] if r['route']==route and r['depth']==depth}),len([r for r in data['sessions'] if r['route']==route and r['depth']==depth]),sum(1 for r in data['sessions'] if r['route']==route and r['depth']==depth and r['passed'])] for route in ['tf','exl3'] for depth in ['8k','64k'] ]),
  data.get('session_grading_note',''),
  '## Broad quality and API compatibility',broad['scope'],table(['Route','Attempted','Passed','API errors','HTTP 400','Grader errors','Truncated'],[[route]+[h.get(k,0) for k in ['attempted','passed','api_errors','http_400','grader_errors','truncated']] for route,h in sorted(broad['health'].items())]),
  table(['Area','Route','Evaluable','Passed','API errors','Grader errors','Truncated'],[[r['area'],r['route'],r['mechanically_evaluable'],r['passed'],r['api_errors'],r['grader_errors'],r['truncated']] for r in categories]),
  table(['Area','Paired outcomes','Distinct items','TF only pass','EXL3 only pass','EXL3 − TF pointwise 95% interval'],[[r['area'],r['paired_outcomes'],r['distinct_items'],r['paired_flips'].get('tf_only',0),r['paired_flips'].get('exl3_only',0),r['item_bootstrap_exl3_minus_tf_pass_fraction_pointwise_95_interval']] for r in broad['paired_comparisons']]),
  '## Regression floor', 'The 40-item mechanical floor is separate from broad quality. Full-answer matching is stricter than the historical first-word grader.', '```json\n'+json.dumps(data['quality'],indent=2)+'\n```',
  '## Failure review']+([failure_review(root)] if (root/'quality-failures.json').exists() else ['No retained quality-failure detail.'])+[
  '## Retrieval and harder code',table(['Route','Input','Rep','Status','All keys correct','Correct keys'],[[r['route'],r['depth'],r['rep'],r['status'],r['passed'],r['correct_keys']] for r in data['retrieval']]),
  data.get('retrieval_grading_note',''),
  table(['Route','Code task','Rep','Passed','Finish','Wall seconds'],[[r['route'],r['item_id'],r['rep'],r['passed'],r['finish_reason'],fmt(r['wall_s'])] for r in data['hard_code']]),
  '## Concurrency','C2/C4 use 32K total context. Aggregate throughput is not single-user decode. Small-sample percentiles are descriptive.','```json\n'+json.dumps(data['concurrency'],indent=2)+'\n```',
  '## Greedy repeatability','Distinct output hashes measure repeatability; cross-arm differences require checking within-arm stability first. EXL3 MTP3 is already known to be non-lossless.','```json\n'+json.dumps(data['parity'],indent=2)+'\n```',
  '## Audit and restoration',f"Excluded speed cells: {len(data['invalid_cells'])}. Prompt-token count mismatches across stacks: {len(data['prompt_token_mismatches'])}; prompt bytes are required to match. Actual counts are retained in speed-cells.csv.",
  'Raw requests, answers, token usage, cache metrics, temperature and energy measurements are retained outside this repository. No canonical benchmark figures or public recipes were changed.',
  'Independent restoration verification: '+('`'+(root/'restoration-verified.json').read_text().strip().replace('\n',' ')+'`' if (root/'restoration-verified.json').exists() else '**Pending.**'),
  'Restoration receipt: '+('`'+(root/'evidence/cleanup.json').read_text().strip()+'`' if (root/'evidence/cleanup.json').exists() else '**Pending; campaign may still be active.**'),
  '## Graphs']
 for name in charts:text.append(f'![{name}]({name}.png)')
 (root/'REPORT.md').write_text('\n\n'.join(text)+'\n')
 # One offline viewer; figures are local files and do not require a server.
 body=''.join('<figure><img style="width:100%;height:auto" src="'+html.escape(name)+'.svg"><figcaption>'+html.escape(name)+'</figcaption></figure>' for name in charts)
 (root/'graphs.html').write_text('<!doctype html><meta charset="utf-8"><title>B70 route comparison</title><style>body{background:#11151b;color:#f2f4f7;font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px}figure{margin:32px 0}p{color:#b8c4d2}</style><h1>B70 route comparison</h1><p>'+html.escape(status)+'</p>'+body)
 print(json.dumps({'report':str(root/'REPORT.md'),'charts':charts,'complete':complete}))

if __name__=='__main__':main()
