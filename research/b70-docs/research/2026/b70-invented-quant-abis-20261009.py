"""E0 CPU-only ABI arithmetic and synthetic round trips; no model conversion."""
import json
import struct


def pack_bits(values, bits):
    x = sum((v & ((1 << bits)-1)) << (i*bits) for i,v in enumerate(values))
    return x.to_bytes((len(values)*bits+7)//8, 'little')


def unpack_bits(data, bits, count):
    x = int.from_bytes(data, 'little')
    return [(x >> (i*bits)) & ((1 << bits)-1) for i in range(count)]

# Canonical INT8 DPAS B-byte logical slot, not unverified compiler GRF ABI.
def slot(n,k):
    return 64*(k//4)+4*n+k%4
assert sorted(slot(n,k) for n in range(16) for k in range(32)) == list(range(512))

# DSM: each K32,N16 tile consists of sign, u, h bitplanes.
for bits in (3,4):
    codes=[i % (1 << bits) for i in range(512)]
    u=[v & ((1 << (bits-2))-1) for v in codes]
    h=[(v >> (bits-2)) & 1 for v in codes]
    sign=[v >> (bits-1) for v in codes]
    planes=[pack_bits(u,bits-2),pack_bits(h,1),pack_bits(sign,1)]
    assert sum(map(len,planes))==512*bits//8
    out=[a+(b << (bits-2))+(c << (bits-1)) for a,b,c in zip(unpack_bits(planes[0],bits-2,512),unpack_bits(planes[1],1,512),unpack_bits(planes[2],1,512))]
    assert out==codes
    # rational scales -> exact two-dot reference vs reconstructed weights
    a=[(i%255)-127 for i in range(512)]
    s,t=3,11
    w=[(-1 if z else 1)*(s*x+t*y) for x,y,z in zip(u,h,sign)]
    assert sum(x*y for x,y in zip(a,w)) == s*sum(x*y*(-1 if z else 1) for x,y,z in zip(a,u,sign))+t*sum(x*y*(-1 if z else 1) for x,y,z in zip(a,h,sign))

# M5X exact doubled E2M1 numeric recode. Signed zero canonicalized by ABI.
mag=[0,1,2,3,4,6,8,12]
v=[(-1 if code&8 else 1)*mag[code&7] for code in range(16)]
raw=v*32
lo=pack_bits([x&15 for x in raw],4)
hi=pack_bits([(x&31)>>4 for x in raw],1)
rec=[a+(b<<4) for a,b in zip(unpack_bits(lo,4,512),unpack_bits(hi,1,512))]
rec=[x-32 if x&16 else x for x in rec]
assert rec==raw and len(lo)+len(hi)==320
assert max(v)>7 and min(v)<-8

# Q3DT: transpose canonical 98-byte blocks, reversible XOR-zero bitmap.
def hot(blocks):
    return bytes(blocks[j][p] for p in range(98) for j in range(32))
def canonical(data):
    return [bytes(data[p*32+j] for p in range(98)) for j in range(32)]
def encode(data):
    delta=bytes(data[p*32+j] ^ (data[p*32+j-1] if j else 0) for p in range(98) for j in range(32))
    mask=pack_bits([bool(v) for v in delta],1)
    sparse=mask+bytes(v for v in delta if v)
    mode=int(len(sparse)<len(data))
    payload=sparse if mode else data
    return struct.pack('<4sBBHII',b'Q3DT',1,mode,32,len(payload),len(data))+payload
def decode(record):
    magic,ver,mode,count,size,rawsize=struct.unpack('<4sBBHII',record[:16])
    assert (magic,ver,count,rawsize)==(b'Q3DT',1,32,3136)
    payload=record[16:]
    assert len(payload)==size
    if not mode: return payload
    mask=unpack_bits(payload[:392],1,3136)
    it=iter(payload[392:]); delta=[next(it) if v else 0 for v in mask]
    out=[]
    for i,d in enumerate(delta): out.append(d ^ (out[-1] if i%32 else 0))
    return bytes(out)
for blocks in ([bytes((p+j)%256 for p in range(98)) for j in range(32)], [bytes(range(98))]*32):
    data=hot(blocks)
    assert canonical(data)==blocks
    assert decode(encode(data))==data

base=4.125
rows=[]
for name,b in [('baseline',base),('DSM3',3.25),('DSM4',4.25),('DSM80_20',3.45),('M5X',5.25),('Q3DT_hot',3.0625),('Q3DT_cold_assumed',2.3125)]:
    d,a=27*b/8,3*b/8
    rows.append(dict(name=name,bpw=b,dense_GB=d,active3B_GB=a,resident30B_GB=30*b/8,delta_percent=100*(b/base-1),dense_pin_ms=1000*d/608,match_baseline430_GBs=430*b/base))
assert 16*8/8192+3.0625*.75==2.3125
print(json.dumps({'status':'E0 synthetic checks PASS; no artifact quality/ISA claim','rows':rows,'KV32K_GB':32768*32768/1e9,'M5X_vs_MXFP4_required_BW_ratio':5.25/4.25,'cold_236MB_at_11_5_GBs_ms':236e6/11.5e9*1000,'cold_assumed_saving_ms':236e6/11.5e9*1000*(1-2.3125/3.0625)},indent=2))
