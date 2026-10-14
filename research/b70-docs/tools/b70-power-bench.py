#!/usr/bin/env python3
"""
B70 Power-Efficiency Benchmark
Measures: tp, tg, ttft, ttlt, wattage, temperature
Finds wattage sweet spot where 10% tg decrease is acceptable
"""

import subprocess
import json
import time
import re
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Model configs (high context 128K focus)
MODELS = [
    {
        "name": "Qwen3.6-35B-A3B-Q4_K_XL",
        "path": "/data/nvme/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        "quant": "Q4_K_XL",
        "type": "MoE",
        "size_gb": 20.6,
        "ctx_targets": [8192, 16384, 32768, 65536, 131072],
    },
    {
        "name": "Ornith-35B-Q4_K_M",
        "path": "/mnt/<models-disk>/B70-Models/ornith-1.0-35b-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "type": "MoE",
        "size_gb": 20.0,
        "ctx_targets": [8192, 16384, 32768, 65536, 131072],
    },
    {
        "name": "Gemma4-26B-A4B-Q4_K_M",
        "path": "/mnt/<models-disk>/B70-Models/gemma-4-26b-A4B-it-UD-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "type": "MoE",
        "size_gb": 15.6,
        "ctx_targets": [8192, 16384, 32768, 65536, 131072],
    },
    {
        "name": "Qwen3.6-27B-Q5_K_M",
        "path": "/mnt/<models-disk>/B70-Models/Qwen3.6-27B-Q5_K_M.gguf",
        "quant": "Q5_K_M",
        "type": "Dense",
        "size_gb": 18.9,
        "ctx_targets": [8192, 16384, 32768, 65536],  # 128K tight on 32GB with q8_0 KV
    },
]

# Power limit targets (in W) - will find sweet spot
# 0 = default (no limit), then test limits
POWER_LIMITS = [0, 250, 225, 200, 175, 150]

LLAMA_CPP_PATH = "/opt/llama.cpp/build-sycl-b70/bin"

PROMPT_TEXT = """In the field of computational complexity theory and algorithmic efficiency, the study of time and space complexity has remained a cornerstone of computer science for decades. From the early work on Turing machines and the Church-Turing thesis to modern developments in quantum computing and neural network complexity, researchers have sought to understand the fundamental limits of what can be computed and how efficiently. The P versus NP problem, one of the seven Millennium Prize Problems, continues to elude resolution despite decades of intense research effort. Meanwhile, advances in approximation algorithms, randomized algorithms, and parameterized complexity have provided practical approaches to tackling computationally hard problems. The relationship between computational models—such as deterministic finite automata, pushdown automata, and Turing machines—and their expressive power has led to deep insights into the nature of computation itself. Furthermore, the rise of machine learning and deep neural networks has raised new questions about computational complexity in the context of high-dimensional optimization and the learnability of complex functions. """ * 10  # ~5K tokens


def get_gpu_metrics() -> Tuple[float, float]:
    """Get current GPU power (W) and temperature (C)"""

    power = 0.0
    temp = 0.0

    try:
        # Try xpu-smi first
        result = subprocess.run(
            ["xpu-smi", "dump"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Parse xpu-smi output
            for line in result.stdout.split("\n"):
                if "Device Power" in line or "Power (W)" in line:
                    m = re.search(r"(\d+\.?\d*)\s*[Ww]?", line)
                    if m:
                        power = float(m.group(1))
                if "Temperature" in line or "Temp (C)" in line:
                    m = re.search(r"(\d+\.?\d*)\s*[°CcC]?", line)
                    if m:
                        temp = float(m.group(1))
            if power > 0 and temp > 0:
                return power, temp

        # Fallback to hwmon
        try:
            temp_path = Path("/sys/class/hwmon/hwmon3/temp1_input")  # B70 hwmon
            if temp_path.exists():
                temp = float(temp_path.read_text().strip()) / 1000.0

            power_path = Path("/sys/class/hwmon/hwmon3/power1_average")
            if power_path.exists():
                # power is in microwatts
                power = float(power_path.read_text().strip()) / 1_000_000.0
                return power, temp
        except Exception:
            pass

        # Second fallback: intel_gpu_top
        result = subprocess.run(
            ["intel_gpu_top", "-l", "1", "-J"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Parse intel_gpu_top JSON
            try:
                data = json.loads(result.stdout)
                # temp and power extraction from JSON
            except:
                pass

    except Exception:
        pass

    return power, temp


def run_llama_bench(
    model_path: str,
    ctx_size: int,
    n_gen: int = 32,
    n_prompt: int = 512,
    power_limit_w: Optional[float] = None,
    n_threads: int = 8,
) -> Dict:
    """Run llama-bench and parse output"""

    env = {
        "ONEAPI_DEVICE_SELECTOR": "level_zero:0",
        "ZES_ENABLE_SYSMAN": "1",
        "SYCL_CACHE_PERSISTENT": "0",
    }

    # Set power limit if specified (needs xpu-smi)
    original_power_limit = None
    if power_limit_w:
        try:
            # Get current power limit
            result = subprocess.run(
                ["xpu-smi", "dump", "-p", "0"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Save for restore (parse from output)
            # Set new limit
            subprocess.run(
                ["xpu-smi", "set-power-limit", "-p", "0", str(power_limit_w)],
                capture_output=True,
                timeout=5,
            )
            time.sleep(1)  # Let it settle
        except Exception as e:
            print(f"[WARN] Failed to set power limit: {e}")

    # Baseline metrics
    power_before, temp_before = get_gpu_metrics()
    time_start = time.time()

    # Run llama-bench
    cmd = [
        f"{LLAMA_CPP_PATH}/llama-bench",
        "-m", model_path,
        "-ngl", "99",
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-r", "1",
        "-c", str(ctx_size),
        "-t", str(n_threads),
        "-o", "md",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        env={**subprocess.os.environ, **env},
    )

    time_total = time.time() - time_start
    power_after, temp_after = get_gpu_metrics()

    # Restore power limit
    if power_limit_w:
        try:
            subprocess.run(
                ["xpu-smi", "set-power-limit", "-p", "0", "default"],
                capture_output=True,
                timeout=5,
            )
        except:
            pass

    # Parse output
    metrics = {
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "time_total_s": time_total,
        "power_w": power_after,
        "temp_c": temp_after,
        "power_delta_w": power_after - power_before,
        "temp_delta_c": temp_after - temp_before,
    }

    # Extract pp and tg speeds from markdown table
    for line in result.stdout.split("\n"):
        if "pp" in line and "t/s" in line:
            m = re.search(r"pp(\d+):\s+(\d+\.?\d*)\s*t/s", line)
            if m:
                metrics["pp_tokens"] = int(m.group(1))
                metrics["tp_tps"] = float(m.group(2))
        if "tg" in line and "t/s" in line:
            m = re.search(r"tg(\d+):\s+(\d+\.?\d*)\s*t/s", line)
            if m:
                metrics["tg_tokens"] = int(m.group(1))
                metrics["tg_tps"] = float(m.group(2))

    # Calculate derived metrics
    if "tp_tps" in metrics:
        metrics["ttft_s"] = metrics["pp_tokens"] / metrics["tp_tps"] if metrics["pp_tokens"] > 0 else 0
    if "tg_tps" in metrics:
        metrics["ttlt_s"] = metrics["tg_tokens"] / metrics["tg_tps"] if metrics["tg_tokens"] > 0 else 0
        metrics["ttft_s"] = metrics.get("ttft_s", 0)

    return metrics


def run_single_model_benchmark(
    model: Dict,
    power_limit_w: Optional[float] = None,
    max_ctx: int = 65536,
) -> List[Dict]:
    """Run benchmark across context sizes for a single model at given power limit"""

    results = []

    model_path = model["path"]
    if not Path(model_path).exists():
        print(f"[SKIP] Model not found: {model_path}")
        return results

    print(f"\n{'='*80}")
    print(f"BENCHMARK: {model['name']} @ {power_limit_w}W limit")
    print(f"{'='*80}")

    for ctx in model["ctx_targets"]:
        if ctx > max_ctx:
            continue

        print(f"\n>>> Context: {ctx:,}")

        try:
            metrics = run_llama_bench(
                model_path=model_path,
                ctx_size=ctx,
                n_gen=32,
                n_prompt=512,
                power_limit_w=power_limit_w,
            )

            result_entry = {
                "model": model["name"],
                "quant": model["quant"],
                "type": model["type"],
                "ctx_size": ctx,
                "power_limit_w": power_limit_w,
                **metrics,
            }

            results.append(result_entry)

            # Print summary
            print(f"  tp:      {metrics.get('tp_tps', 0):.2f} t/s")
            print(f"  tg:      {metrics.get('tg_tps', 0):.2f} t/s")
            print(f"  ttft:    {metrics.get('ttft_s', 0):.3f} s")
            print(f"  ttlt:    {metrics.get('ttlt_s', 0):.3f} s")
            print(f"  power:   {metrics.get('power_w', 0):.1f} W")
            print(f"  temp:    {metrics.get('temp_c', 0):.1f} C")
            print(f"  exit:    {metrics['exit_code']}")

            # Sleep between runs to let temps normalize
            time.sleep(10)

        except subprocess.TimeoutExpired:
            print(f"  [ERROR] Timeout after 300s")
        except Exception as e:
            print(f"  [ERROR] {e}")

    return results


def find_power_sweet_spot(
    model: Dict,
    target_ctx: int = 32768,
) -> Tuple[Optional[float], float, Dict]:
    """
    Find power limit sweet spot where 10% tg decrease is acceptable.
    Returns: (best_power_limit_w, efficiency_score, all_results)
    """

    print(f"\n{'='*80}")
    print(f"FINDING POWER SWEET SPOT: {model['name']}")
    print(f"Target context: {target_ctx}")
    print(f"{'='*80}")

    all_results = []

    # Get baseline (no power limit)
    baseline_results = run_single_model_benchmark(
        model, power_limit_w=None, max_ctx=target_ctx
    )
    if not baseline_results:
        print("[ERROR] No baseline results")
        return None, 0.0, {}

    baseline = baseline_results[0]
    baseline_tg = baseline.get("tg_tps", 0)
    baseline_power = baseline.get("power_w", 0)

    print(f"\nBaseline: tg={baseline_tg:.2f} t/s @ {baseline_power:.1f} W")

    # Test power limits
    power_results = {}
    for power_limit in POWER_LIMITS:
        if power_limit is None:
            continue  # Skip baseline (already have it)

        results = run_single_model_benchmark(
            model, power_limit_w=power_limit, max_ctx=target_ctx
        )
        if results:
            power_results[power_limit] = results[0]

    # Find sweet spot
    best_power = None
    best_score = 0.0
    best_result = None

    for power_limit, result in power_results.items():
        tg = result.get("tg_tps", 0)
        power = result.get("power_w", 0)

        if tg == 0:
            continue

        # Efficiency score: (tg / baseline_tg) * (baseline_power / power)
        # We want high tg but low power
        tg_ratio = tg / baseline_tg if baseline_tg > 0 else 0
        power_ratio = baseline_power / power if power > 0 else 0

        # Sweet spot: accept 10% tg decrease (0.9 ratio)
        if tg_ratio < 0.9:
            print(f"\n{power_limit}W: REJECTED (tg only {tg_ratio*100:.1f}% of baseline)")
            continue

        score = tg_ratio * power_ratio
        print(
            f"\n{power_limit}W: tg={tg:.2f} ({tg_ratio*100:.1f}%) power={power:.1f} ({power_ratio*100:.1f}%) score={score:.3f}"
        )

        if score > best_score:
            best_score = score
            best_power = power_limit
            best_result = result

    print(f"\n{'='*80}")
    print(f"RECOMMENDED POWER LIMIT: {best_power}W")
    print(f"Efficiency score: {best_score:.3f}")
    print(f"{'='*80}")

    return best_power, best_score, {"baseline": baseline, "limits": power_results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name (use all if not specified)")
    parser.add_argument("--power-sweep", action="store_true", help="Run power sweet spot finder")
    parser.add_argument("--ctx-limit", type=int, default=65536, help="Max context to test")
    parser.add_argument("--output", type=str, default="/tmp/b70-benchmark-results.json", help="Output file")
    args = parser.parse_args()

    all_results = []

    # Check xpu-smi
    try:
        subprocess.run(["xpu-smi", "version"], capture_output=True, timeout=5, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[WARN] xpu-smi not available - power/temp metrics may fail")

    # Select models
    models_to_test = MODELS
    if args.model:
        models_to_test = [m for m in MODELS if args.model.lower() in m["name"].lower()]

    for model in models_to_test:
        if args.power_sweep:
            # Run power sweet spot finder
            best_power, score, results = find_power_sweet_spot(model, target_ctx=args.ctx_limit)
            if results:
                all_results.append(
                    {
                        "mode": "power_sweep",
                        "model": model["name"],
                        "best_power_w": best_power,
                        "efficiency_score": score,
                        "results": results,
                    }
                )
        else:
            # Run full context sweep at default power
            results = run_single_model_benchmark(model, power_limit_w=None, max_ctx=args.ctx_limit)
            all_results.extend(results)

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()