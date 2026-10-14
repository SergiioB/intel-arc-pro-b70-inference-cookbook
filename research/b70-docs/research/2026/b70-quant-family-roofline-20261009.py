"""CPU-only E0 weight-streaming ceilings, not benchmark predictions."""
import json
from pathlib import Path
formats = [
('GPTQ sym G128',4.125),('GPTQ sym G64',4.25),('IQ4_XS',4.25),
('IQ2_XXS',2.0625),('IQ2_XS',2.3125),('IQ2_S',2.5625),
('IQ1_S',1.5625),('IQ1_M',1.75),('TQ1_0',1.6875),('TQ2_0',2.0625),
('MXFP4',4.25),('MXFP8',8.25),('NVFP4',4.5),
('Q4_K',4.5),('Q5_K',5.5),('Q6_K',6.5625),('Q8_0',8.5),
('EXL3 4',4),('EXL3 3',3),('EXL3 2.5',2.5),('EXL3 2',2),
('lattice 2 payload',2),('lattice 3 payload',3),('IQ3_XXS',3.0625),('IQ3_S',3.4375),
('affine W2 G128 sym',2.125),('affine W3 G128 sym',3.125)]
rows=[]
for name,b in formats:
    d,a,r=27*b/8,3*b/8,30*b/8
    row=dict(name=name,bpw=b,dense_GB=d,moe_active_GB=a,moe_resident_GB=r,dense_y1=608/d,dense_y3=3*608/d,moe_y1=608/a,moe_y3=3*608/a)
    rows.append(row)
    print(f'| {name} | {b:g} | {d:.3f} / {a:.3f} | {608/d:.1f} / {1824/d:.1f} | {608/a:.0f} / {1824/a:.0f} |')
print('CHECKS',json.dumps({'trellis_instr_per_byte':336/512,'trellis_Ginstr_27B_4':13.5*336/512,'pin_fraction':381/608,'ablation_loss_approx':1-381/636,'dense_4_y1_at381':381/13.5,'dense_4_y3_at381':3*381/13.5,'bf16_dense_compute_tps':170e12/(2*27e9),'int8_dense_compute_tps':363e12/(2*27e9),'bf16_moe_compute_tps':170e12/(2*3e9),'int8_moe_compute_tps':363e12/(2*3e9),'fp16_ridge_M_4bit':(170e12/608e9)/4,'int8_ridge_M_4bit':(363e12/608e9)/4},indent=2))
print('CONDITIONAL_ROUND (not uniform weight-only)')
for b in (4,3,2.5,2):
    d=12.230*b/4+4.450
    print(dict(body_bpw=b,round_GB=d,y1=608/d,y3=1824/d,y265=2.65*608/d))
assert abs(rows[0]['dense_GB']-13.921875)<1e-9
assert max(abs(r['dense_y3']/r['dense_y1']-3) for r in rows)<1e-12
