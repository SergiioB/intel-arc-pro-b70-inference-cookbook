#!/bin/bash
for hw in /sys/class/drm/card*/device/hwmon/hwmon*/; do
  [ -d "$hw" ] || continue
  t=$(cat "$hw"/temp1_input 2>/dev/null || cat "$hw"/temp2_input 2>/dev/null)
  p=$(cat "$hw"/power1_average 2>/dev/null)
  [ -n "$t" ] && echo "temp_C=$((t/1000)) power_mW=$((p/1000)) hwmon=$hw"
done
if command -v xpu-smi >/dev/null 2>&1; then xpu-smi stats -d 0 2>/dev/null | head -20; fi
