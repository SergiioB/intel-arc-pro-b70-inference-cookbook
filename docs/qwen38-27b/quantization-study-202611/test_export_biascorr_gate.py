#!/usr/bin/env python3
"""CPU tests for the KL gate in export_biascorr.py. Run: python3 test_export_biascorr_gate.py -v

Needs torch, transformers (with Qwen3_5), safetensors, tokenizers and numpy. No GPU.
"""
import contextlib
import io
import json
import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig

import export_biascorr as eb

HERE = os.path.dirname(os.path.abspath(__file__))
RECEIPT_15A = os.path.join(HERE, "receipts", "exp15a_tfq3_vs_hfg16_64l.json")


def arms(base_mean, plus_shift, minus_shift, seed, noise=0.002):
    """Per-position KL for three arms, shape (4, 511), with exact mean shifts."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(2.0, base_mean / 2.0, size=(4, 511))

    def shifted(shift):
        e = rng.normal(0.0, noise, size=base.shape)
        return base + shift + e - e.mean()

    return {"base": base, "plus": shifted(plus_shift), "minus": shifted(minus_shift)}


class GateLogic(unittest.TestCase):
    def test_pass(self):
        out = eb.gate_sidecar({"prose": arms(0.0506, -0.0116, +0.0365, 1), "code": arms(0.184, -0.0253, +0.114, 2)})
        self.assertEqual(out["status"], "PUBLISHABLE")
        self.assertEqual(out["failed_checks"], [])
        for dom in ("prose", "code"):
            self.assertLess(out["domains"][dom]["delta_plus_ci95_blk32"][1], 0)

    def test_gptq_like_prose_regression_is_rejected(self):
        # Shifts copied from exp13a: +b hurts GPTQ on prose, helps on code.
        out = eb.gate_sidecar({"prose": arms(0.0494, +0.00279, +0.00415, 3), "code": arms(0.302, -0.01498, +0.0383, 4)})
        self.assertEqual(out["status"], "REJECTED_BY_KL_GATE")
        self.assertEqual(out["failed_checks"], ["prose_kl_drops_ci95_below_0"])
        self.assertGreater(out["domains"]["prose"]["delta_plus_ci95_blk32"][0], 0)

    def test_code_regression_is_rejected(self):
        out = eb.gate_sidecar({"prose": arms(0.05, -0.01, +0.03, 5), "code": arms(0.2, +0.02, +0.05, 6)})
        self.assertEqual(out["failed_checks"], ["code_kl_drops_ci95_below_0"])

    def test_sign_flip_that_helps_is_rejected_per_domain(self):
        good = arms(0.05, -0.01, +0.03, 7)
        for bad_domain in ("prose", "code"):
            kl = {"prose": good, "code": good}
            kl[bad_domain] = arms(0.05, -0.01, -0.005, 8)
            out = eb.gate_sidecar(kl)
            self.assertEqual(out["status"], "REJECTED_BY_KL_GATE")
            self.assertEqual(out["failed_checks"], [f"{bad_domain}_minus_b_worsens_kl"])

    def test_mean_drop_inside_the_noise_is_rejected(self):
        noisy = arms(0.05, -0.0002, +0.03, 9, noise=0.05)
        out = eb.gate_sidecar({"prose": noisy, "code": arms(0.2, -0.02, +0.05, 10)})
        d = out["domains"]["prose"]
        self.assertLess(d["delta_plus"], 0)
        self.assertGreater(d["delta_plus_ci95_blk32"][1], 0)
        self.assertEqual(out["failed_checks"], ["prose_kl_drops_ci95_below_0"])

    def test_bad_input_gives_no_verdict(self):
        good = arms(0.05, -0.01, +0.03, 11)
        nan = {k: v.copy() for k, v in good.items()}
        nan["plus"][0, 0] = np.nan
        one_block = {k: v[:, :32] for k, v in good.items()}
        flat = {k: v.reshape(-1) for k, v in good.items()}
        uneven = dict(good, minus=good["minus"][:, :-1])
        no_minus = {k: good[k] for k in ("base", "plus")}
        for bad in (nan, one_block, flat, uneven, no_minus):
            with self.assertRaises(ValueError):
                eb.gate_sidecar({"prose": bad, "code": good})
        with self.assertRaises(ValueError):
            eb.gate_sidecar({"prose": good})

    @unittest.skipUnless(os.path.exists(RECEIPT_15A), "study receipt not shipped")
    def test_bootstrap_reproduces_exp15a_receipt(self):
        with open(RECEIPT_15A) as f:
            rec = json.load(f)
        pair = rec["paired_blk32_bootstrap_95ci"]["TFQ3_4b_matched400_plus_BC__vs__TFQ3_4b_matched400"]
        for dom in ("wt2", "code"):
            per = rec["per_position"][dom]["kl"]
            delta = np.asarray(per["TFQ3_4b_matched400_plus_BC"]) - np.asarray(per["TFQ3_4b_matched400"])
            _, lo, hi = eb.blk32_ci(delta.reshape(4, 511))
            np.testing.assert_allclose([lo, hi], pair[dom]["delta_kl"]["ci95_blk32_within_window"], atol=2e-6)


# ---------------------------------------------------------------------------
# End to end: the real CLI on a tiny Qwen3.5 checkpoint pair, on CPU.
# ---------------------------------------------------------------------------
LAYERS = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
TEXT_CONFIG = dict(
    hidden_size=128, num_hidden_layers=4, layer_types=LAYERS, num_attention_heads=2, num_key_value_heads=2,
    head_dim=64, intermediate_size=192, vocab_size=256, linear_num_key_heads=2, linear_num_value_heads=2,
    linear_key_head_dim=64, linear_value_head_dim=64, linear_conv_kernel_dim=4, tie_word_embeddings=False,
)


def quantize_mlx(w):
    """MLX affine uint4, group 64: returns (packed u32, scales, biases, dequantized BF16)."""
    rows, k = w.shape
    g = w.float().reshape(rows, k // 64, 64)
    lo, hi = g.amin(-1), g.amax(-1)
    s = ((hi - lo) / 15).clamp(min=1e-6).to(torch.bfloat16)
    b = lo.to(torch.bfloat16)
    q = ((g - b.float().unsqueeze(-1)) / s.float().unsqueeze(-1)).round().clamp(0, 15)
    deq = (q * s.float().unsqueeze(-1) + b.float().unsqueeze(-1)).reshape(rows, k).to(torch.bfloat16)
    nib = q.to(torch.int64).reshape(rows, k // 8, 8)
    packed = (nib << torch.arange(0, 32, 4)).sum(-1).to(torch.uint32)
    return packed, s.contiguous(), b.contiguous(), deq


def save_checkpoint(path, tensors):
    os.makedirs(path, exist_ok=True)
    save_file(tensors, os.path.join(path, "model.safetensors"))
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: "model.safetensors" for k in tensors}}, f)


def build_tiny(root, lossless):
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(**TEXT_CONFIG)
    model = Qwen3_5ForCausalLM(cfg)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # A strong shared direction gives every linear a large mean input, which is what bias correction fixes.
    emb = state["model.embed_tokens.weight"]
    state["model.embed_tokens.weight"] = 0.02 * torch.randn_like(emb) + 0.25 * torch.randn(1, emb.shape[1])
    state["lm_head.weight"] = 0.5 * torch.randn_like(state["lm_head.weight"])
    bf, mlx = {}, {}
    quantized = {name + ".weight" for name, m in model.named_modules() if isinstance(m, nn.Linear)} | {"model.embed_tokens.weight"}
    for key, value in state.items():
        value = value.to(torch.bfloat16).contiguous()
        bf_key = key if key.startswith("lm_head") else "model.language_model." + key[len("model."):]
        if key in quantized:
            stem = "language_model.lm_head" if key.startswith("lm_head") else "language_model.model." + key[len("model."):-len(".weight")]
            packed, s, b, deq = quantize_mlx(value)
            mlx[stem + ".weight"], mlx[stem + ".scales"], mlx[stem + ".biases"] = packed, s, b
            if lossless:
                value = deq.contiguous()
        bf[bf_key] = value
    bf_dir, mlx_dir = os.path.join(root, "bf16"), os.path.join(root, "mlx")
    save_checkpoint(bf_dir, bf)
    save_checkpoint(mlx_dir, mlx)
    with open(os.path.join(bf_dir, "config.json"), "w") as f:
        json.dump({"text_config": TEXT_CONFIG}, f)
    vocab = {"[UNK]": 0, **{f"w{i}": i for i in range(1, 256)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]").save_pretrained(bf_dir)
    rng = np.random.default_rng(0)
    texts = {}
    for name in ("cal_prose", "cal_code", "eval_prose", "eval_code"):
        texts[name] = os.path.join(root, name + ".txt")
        with open(texts[name], "w") as f:
            f.write(" ".join(f"w{i}" for i in rng.integers(1, 256, size=600)))
    return bf_dir, mlx_dir, texts


def run_cli(bf_dir, mlx_dir, out, texts, *extra, gate=True):
    argv = ["--bf16-dir", bf_dir, "--mlx-dir", mlx_dir, "--out", out, "--device", "cpu", "--seq-len", "96",
            "--prose-files", texts["cal_prose"], "--code-files", texts["cal_code"]]
    if gate:
        argv += ["--eval-prose", texts["eval_prose"], "--eval-code", texts["eval_code"], "--eval-seqs", "2"]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return eb.main(argv + list(extra))


def read_meta(out):
    with open(os.path.splitext(out)[0] + ".meta.json") as f:
        return json.load(f)


def full_model_kl(bf_dir, mlx_dir, sidecar, rows, sign):
    """Independent oracle: ordinary transformers forward passes, no layer streaming."""
    bf = load_file(os.path.join(bf_dir, "model.safetensors"))
    mlx = load_file(os.path.join(mlx_dir, "model.safetensors"))
    bias = load_file(sidecar)
    cfg = Qwen3_5TextConfig(**TEXT_CONFIG)

    def model_with(state):
        m = Qwen3_5ForCausalLM(cfg).eval()
        # Cast weights only. Module.to(bfloat16) would also round the rotary frequencies.
        for param in m.parameters():
            param.data = param.data.to(torch.bfloat16)
        m.load_state_dict(state, strict=True)
        return m

    ref_state = {k.replace("model.language_model.", "model."): v for k, v in bf.items()}
    q_state = dict(ref_state)
    for key in mlx:
        if key.endswith(".scales"):
            stem = key[:-len(".scales")]
            hf = "lm_head.weight" if stem.endswith("lm_head") else "model." + stem[len("language_model.model."):] + ".weight"
            q_state[hf] = eb.unpack_mlx_u32_bf16(mlx[stem + ".weight"].view(torch.int32), mlx[stem + ".scales"], mlx[stem + ".biases"])
    ref, quant = model_with(ref_state), model_with(q_state)
    if sign:
        for name, mod in quant.named_modules():
            key = "language_model." + name + ".bias_corr"
            if key in bias:
                mod.register_forward_hook(lambda _m, _i, o, b=sign * bias[key]: (o.float() + b).to(o.dtype))
    with torch.no_grad():
        ids = torch.tensor(rows)
        lp = F.log_softmax(ref(input_ids=ids, use_cache=False).logits[:, :-1].float(), -1)
        lq = F.log_softmax(quant(input_ids=ids, use_cache=False).logits[:, :-1].float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1).clamp(min=0).numpy()


class CliEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.lossy = build_tiny(os.path.join(cls.tmp.name, "lossy"), lossless=False)
        cls.lossless = build_tiny(os.path.join(cls.tmp.name, "lossless"), lossless=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_real_bias_passes_and_matches_full_model_forward(self):
        bf_dir, mlx_dir, texts = self.lossy
        out = os.path.join(self.tmp.name, "pass.safetensors")
        self.assertEqual(run_cli(bf_dir, mlx_dir, out, texts), 0)
        meta = read_meta(out)
        self.assertEqual(meta["status"], "PUBLISHABLE", meta["kl_gate"])
        tok = eb.AutoTokenizer.from_pretrained(bf_dir)
        windows = {d: eb.tokenize_windows(tok, [texts["eval_" + d]], 96, 2) for d in ("prose", "code")}
        kl = eb.eval_causal_kl(bf_dir, mlx_dir, out, windows, torch.device("cpu"))
        for dom in ("prose", "code"):
            for arm, sign in (("base", 0), ("plus", 1), ("minus", -1)):
                oracle = full_model_kl(bf_dir, mlx_dir, out, windows[dom], sign)
                np.testing.assert_allclose(kl[dom][arm], oracle, rtol=1e-3, atol=1e-5, err_msg=f"{dom} {arm}")
            self.assertAlmostEqual(meta["kl_gate"]["domains"][dom]["kl_base"], kl[dom]["base"].mean(), places=9)

    def test_useless_bias_exits_3_and_is_marked_rejected(self):
        # BF16 equals the dequantized MLX weights, so there is nothing to correct.
        bf_dir, mlx_dir, texts = self.lossless
        out = os.path.join(self.tmp.name, "reject.safetensors")
        self.assertEqual(run_cli(bf_dir, mlx_dir, out, texts, "--kl-gate"), 3)
        meta = read_meta(out)
        self.assertEqual(meta["status"], "REJECTED_BY_KL_GATE")
        self.assertEqual(len(meta["kl_gate"]["failed_checks"]), 4)

    def test_without_the_gate_the_sidecar_is_ungated_with_identical_tensors(self):
        bf_dir, mlx_dir, texts = self.lossy
        plain = os.path.join(self.tmp.name, "plain.safetensors")
        gated = os.path.join(self.tmp.name, "gated.safetensors")
        self.assertEqual(run_cli(bf_dir, mlx_dir, plain, texts, "--meta-out", plain + ".json", gate=False), 0)
        with open(plain + ".json") as f:
            self.assertEqual(json.load(f)["status"], "UNGATED")
        run_cli(bf_dir, mlx_dir, gated, texts)
        # Compare tensors: safetensors writes metadata keys in a random order, so file hashes differ.
        a, b = load_file(plain), load_file(gated)
        self.assertEqual(a.keys(), b.keys())
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)
        # --no-kl-gate wins even when eval files are present.
        self.assertEqual(run_cli(bf_dir, mlx_dir, gated, texts, "--no-kl-gate", "--meta-out", plain + ".json"), 0)
        with open(plain + ".json") as f:
            self.assertEqual(json.load(f)["status"], "UNGATED")

    def test_crashed_gate_overwrites_an_old_pass(self):
        bf_dir, mlx_dir, texts = self.lossy
        out = os.path.join(self.tmp.name, "crash.safetensors")
        self.assertEqual(run_cli(bf_dir, mlx_dir, out, texts), 0)
        self.assertEqual(read_meta(out)["status"], "PUBLISHABLE")
        real = eb.eval_causal_kl

        def boom(*_a, **_k):
            raise RuntimeError("device lost")

        eb.eval_causal_kl = boom
        try:
            with self.assertRaises(RuntimeError):
                run_cli(bf_dir, mlx_dir, out, texts)
        finally:
            eb.eval_causal_kl = real
        self.assertEqual(read_meta(out)["status"], "GATE_PENDING")

    def test_bad_flag_combinations_stop_before_any_output(self):
        bf_dir, mlx_dir, texts = self.lossy
        out = os.path.join(self.tmp.name, "never.safetensors")
        for extra, gate in ((["--kl-gate"], False), (["--eval-prose", texts["eval_prose"]], False), (["--zero-bias"], True),
                            (["--eval-prose", texts["cal_prose"]], True)):
            with self.assertRaises(SystemExit) as ctx:
                run_cli(bf_dir, mlx_dir, out, texts, *extra, gate=gate)
            self.assertEqual(ctx.exception.code, 2, extra)
            self.assertFalse(os.path.exists(out), extra)


if __name__ == "__main__":
    unittest.main()
