---
title: "Run-4 subagent report: harness-sc-copy-rca (2026-10-12)"
type: research
date: 2026-10-12
status: provisional
---

# SC copy-engine RCA (CPU-only)

Pre-fix source lifetime/thread inventory:
```json
[
  {
    "line": 165,
    "text": "signs = (torch.randint(0, 2, (n,), generator=gen) * 2 - 1).to(dev, torch.float32)"
  },
  {
    "line": 357,
    "text": "return Lb.to(device=C.device, dtype=C.dtype), Li.to(device=C.device, dtype=C.dtype)"
  },
  {
    "line": 404,
    "text": "del Lb, Li"
  },
  {
    "line": 405,
    "text": "del sc"
  },
  {
    "line": 423,
    "text": "del L, eye"
  },
  {
    "line": 466,
    "text": "del diag"
  },
  {
    "line": 502,
    "text": "U = U_cpu.float().to(dev_h)"
  },
  {
    "line": 503,
    "text": "del U_cpu"
  },
  {
    "line": 509,
    "text": "U2 = hinv_upper_inplace(H_cpu.to(dev_h))"
  },
  {
    "line": 512,
    "text": "U2 = U2_cpu.float().to(dev_h)"
  },
  {
    "line": 513,
    "text": "del U2_cpu"
  },
  {
    "line": 520,
    "text": "del U2"
  },
  {
    "line": 523,
    "text": "U_ref = U_ref_cpu.float().to(dev_h)"
  },
  {
    "line": 524,
    "text": "del U_ref_cpu"
  },
  {
    "line": 529,
    "text": "del U_ref"
  },
  {
    "line": 540,
    "text": "del H_cpu"
  },
  {
    "line": 599,
    "text": "del layer"
  },
  {
    "line": 654,
    "text": "sd = {k: v.to(dev) for k, v in sd.items()}"
  },
  {
    "line": 655,
    "text": "Q = H.to(dev).float()"
  },
  {
    "line": 703,
    "text": "dev is not None: the integer trellis reconstruction (ref.reconstruct_inner: pure"
  },
  {
    "line": 731,
    "text": "inner = ref.reconstruct_inner(tre.to(dev), k_bits, cb).cpu()"
  },
  {
    "line": 733,
    "text": "inner_ref = ref.reconstruct_inner(tre, k_bits, cb)"
  },
  {
    "line": 739,
    "text": "del inner_ref, w_ref"
  },
  {
    "line": 747,
    "text": "del inner"
  },
  {
    "line": 749,
    "text": "del tre, suh, svh, w_in_out"
  },
  {
    "line": 771,
    "text": "inner = ref.reconstruct_inner(block.to(dev), K, cb).cpu()"
  },
  {
    "line": 776,
    "text": "expected = ref.reconstruct_inner(sample, K, cb)"
  },
  {
    "line": 777,
    "text": "actual = ref.reconstruct_inner(sample.to(dev), K, cb).cpu()"
  },
  {
    "line": 785,
    "text": "del expected, actual"
  },
  {
    "line": 786,
    "text": "weight_gpu = weight.to(torch.bfloat16).to(dev).float()"
  },
  {
    "line": 788,
    "text": "del block, inner, weight, weight_gpu"
  },
  {
    "line": 908,
    "text": "{k: v.float().to(dev) for k, v in sd_cpu.items()},"
  },
  {
    "line": 911,
    "text": "del O"
  },
  {
    "line": 912,
    "text": "torch.xpu.empty_cache()"
  },
  {
    "line": 915,
    "text": "del sd_fold32"
  },
  {
    "line": 917,
    "text": "del R"
  },
  {
    "line": 918,
    "text": "torch.xpu.empty_cache()"
  },
  {
    "line": 948,
    "text": "{k: v.float().to(dev) for k, v in sd_cpu.items()},"
  },
  {
    "line": 951,
    "text": "del O"
  },
  {
    "line": 952,
    "text": "torch.xpu.empty_cache()"
  },
  {
    "line": 958,
    "text": "del d, yR32"
  },
  {
    "line": 959,
    "text": "torch.xpu.empty_cache()"
  },
  {
    "line": 973,
    "text": "del yOb, yRb, yO32"
  },
  {
    "line": 974,
    "text": "torch.xpu.empty_cache()"
  },
  {
    "line": 991,
    "text": "e_ev = emb[ev.to(dev)].to(self.dtype)"
  },
  {
    "line": 992,
    "text": "e_can = [emb[c[\"ids\"].to(dev)].to(self.dtype) for c in canaries]"
  },
  {
    "line": 993,
    "text": "e_cal = (emb[calib.to(dev)].to(self.dtype)"
  },
  {
    "line": 1006,
    "text": "del e_ev, e_can, e_cal"
  },
  {
    "line": 1007,
    "text": "torch.xpu.empty_cache()"
  },
  {
    "line": 1023,
    "text": "torch.xpu.empty_cache()"
  },
  {
    "line": 1028,
    "text": "Worig = sd_cpu[name + \".weight\"].to(dev, torch.float32)"
  },
  {
    "line": 1035,
    "text": "gain = 1.0 + sd_cpu[norm].to(dev, torch.float32)"
  },
  {
    "line": 1074,
    "text": "H = Hc.to(dev)"
  },
  {
    "line": 1079,
    "text": "del L, eye"
  },
  {
    "line": 1082,
    "text": "del X"
  },
  {
    "line": 1086,
    "text": "del M"
  },
  {
    "line": 1098,
    "text": "torch.xpu.empty_cache()"
  },
  {
    "line": 1101,
    "text": "del Hc, last_u"
  },
  {
    "line": 1179,
    "text": "cpu = ref.reconstruct_inner(tre, K, ref.CB_MUL1)"
  },
  {
    "line": 1180,
    "text": "xpu = ref.reconstruct_inner(tre.to(dev), K, ref.CB_MUL1).cpu()"
  },
  {
    "line": 1187,
    "text": "del tre, cpu, xpu"
  },
  {
    "line": 1211,
    "text": "Q = build_H_5120(SEED).to(dev)"
  },
  {
    "line": 1216,
    "text": "del Q64"
  },
  {
    "line": 1233,
    "text": "emb = f.get_tensor(emb_key).to(dev)"
  },
  {
    "line": 1236,
    "text": "e = artifact_tensor(EXL3_ROOTS[n], emb_key).to(dev) if n in EXL3_ARMS and not CALIB_ARMS else emb"
  },
  {
    "line": 1240,
    "text": "del e"
  },
  {
    "line": 1241,
    "text": "del emb"
  },
  {
    "line": 1252,
    "text": "from concurrent.futures import ThreadPoolExecutor"
  },
  {
    "line": 1253,
    "text": "ex_pool = ThreadPoolExecutor(max_workers=1)"
  },
  {
    "line": 1256,
    "text": "def _exl3_prefetch(Lp, arm):"
  },
  {
    "line": 1263,
    "text": "ex_fut = {(layers[0], a): ex_pool.submit(_exl3_prefetch, layers[0], a) for a in EXL3_ARMS}"
  },
  {
    "line": 1279,
    "text": "built[\"bf16\"] = hst.build_layer(txt_cfg, L, sd_cpu).to(dev)"
  },
  {
    "line": 1290,
    "text": "del sd_fold32"
  },
  {
    "line": 1304,
    "text": "built[a] = hst.build_layer(txt_cfg, L, sd_ex).to(dev)"
  },
  {
    "line": 1306,
    "text": "del sd_ex, ex_w"
  },
  {
    "line": 1308,
    "text": "ex_fut[(layers[li + 1], a)] = ex_pool.submit(_exl3_prefetch, layers[li + 1], a)"
  },
  {
    "line": 1309,
    "text": "del sd_fold"
  },
  {
    "line": 1337,
    "text": "del XtX"
  },
  {
    "line": 1361,
    "text": "del mods_"
  },
  {
    "line": 1399,
    "text": "del store, probes"
  },
  {
    "line": 1423,
    "text": "del H_reference"
  },
  {
    "line": 1424,
    "text": "del reference_mods"
  },
  {
    "line": 1440,
    "text": "del H, Us, hi"
  },
  {
    "line": 1452,
    "text": "W[:, dm.to(dev)] = 0.0"
  },
  {
    "line": 1454,
    "text": "del W"
  },
  {
    "line": 1461,
    "text": "del q, c, ss"
  },
  {
    "line": 1468,
    "text": "del Href, diag_ref"
  },
  {
    "line": 1469,
    "text": "Uref = Uref_cpu.float().to(dev)"
  },
  {
    "line": 1470,
    "text": "del Uref_cpu"
  },
  {
    "line": 1479,
    "text": "del Uref, qr, cr, fp32_weight, fp64_weight"
  },
  {
    "line": 1484,
    "text": "uc = uf[\"U_classic_fp64\"].to(dev)"
  },
  {
    "line": 1490,
    "text": "del uf, uc, cc"
  },
  {
    "line": 1500,
    "text": "del W, Wq, codes, scales"
  },
  {
    "line": 1501,
    "text": "del U, H_calib"
  },
  {
    "line": 1504,
    "text": "torch.xpu.empty_cache()"
  },
  {
    "line": 1508,
    "text": "del mods, Hacc"
  },
  {
    "line": 1543,
    "text": "del reference_state"
  },
  {
    "line": 1545,
    "text": "del built"
  },
  {
    "line": 1546,
    "text": "torch.xpu.empty_cache()"
  },
  {
    "line": 1571,
    "text": "w_norm = f.get_tensor(norm_key).to(dev)"
  },
  {
    "line": 1574,
    "text": "lm_head = f.get_tensor(head_key).to(dev)          # [vocab, 5120] bf16"
  },
  {
    "line": 1582,
    "text": "norm = artifact_tensor(EXL3_ROOTS[arm.name], norm_key).to(dev) if arm.name in EXL3_ARMS and not CALIB_ARMS else w_norm"
  },
  {
    "line": 1603,
    "text": "ev_ids = eval_ids.to(dev)"
  },
  {
    "line": 1625,
    "text": "score_mask = build_inputs.response_mask.to(dev).reshape(-1)"
  },
  {
    "line": 1648,
    "text": "nll_mask = build_inputs.response_mask[:, :-1].to(dev).reshape(-1)"
  },
  {
    "line": 1650,
    "text": "del logits[a]"
  },
  {
    "line": 1651,
    "text": "torch.xpu.empty_cache()"
  }
]
```

No non_blocking argument occurs. Worker XPU call chain: _exl3_prefetch1256 -> decode_exl3_block1259 -> reconstruct_inner(tre.to(dev))731. Main embedding copies1233/1236; Arm.__init__ deletes1006 and empty_cache1007. Important temporal constraint: executor creation1253 occurs AFTER all embedding Arm constructors1240/1241 in this actual source, so simultaneous embedding copies/worker execution is not established by source ordering. Cross-thread device use definitely occurs later in layer build/decode/prefetch.

## CPU-only fix and verification
- CPU-only fileprefetch `load_exl3_block_cpu`, no ref/XPU calls; `decode_exl3_block` enforces main-threadXPU and executes afterfuture.result.
- Explicit CPUembeddingcopy sources retaineduntilsynchronize; artifactcopiesandArmconstructionserializedwithbarriers.
- All11empty_cache sites directlyprecededbysynchronize. CPUtestactualSCqkvpacked tensors deviceCPU4/4 with xpu sync/cachemonkeypatchedraise; nohit. Sourcecheck11/11cachebarriers,no non_blocking.
- Originalsourceordering executorcreatedAFTERembeddings,soexactembedding-workerconcurrencyunproven. Genericallocatorfreerace+latercross-threadXPUremainleadinghypothesis,notrootcauseproven.
- Saveddevcoredump508712bytesSHA96759ad73457841f3a041f8f6e33cd14cac7dea8e04ffe37f5faa9552a7bfaf5;captureReasonTimedoutjob,python3PID3209552,bcs6.
- MainauthorizedONErerunafterfix; projection[INFERENCE]40-60min4arm64; noGPUusedduringRCA.
