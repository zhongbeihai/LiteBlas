#!/usr/bin/env bash
# grade.sh — run ./mm for many sizes; average the last numeric column
# robust: does NOT exit early on failures

set -u   # keep undefined-var safety; avoid -e and pipefail so we don't abort the loop

KERNEL="${KERNEL:-my_kernel}"

read -r -d '' SIZES_STR <<'EOF'
32 64 128 255 256 510 512 513 768 769 1023 1024 1025 1033 2047 2048 2049
EOF

MIN_SIZE=512

# Turn the multi-line string into a bash array (newlines/spaces are fine)
# shellcheck disable=SC2206
SIZES=($SIZES_STR)

echo "Kernel: ${KERNEL}"
echo "Sizes: ${#SIZES[@]}"
echo

sum="0"
count=0
fail=0

for sz in "${SIZES[@]}"; do
  echo "==> ./mm --kernel=${KERNEL} --size=${sz} --reps=10"
  output=$(./mm --kernel="${KERNEL}" --size="${sz}" --reps=10 2>&1)
  status=$?

  # show the program's last non-empty line (the performance line)
  last_line=$(printf "%s\n" "$output" | awk 'NF{line=$0} END{print line}')
  echo "   $last_line"

  if (( status != 0 )); then
    echo "   [error] exit ${status} for size ${sz} — continuing"
    ((fail++))
    continue
  fi

  # last field is the number (e.g., 6.7475)
  val=$(printf "%s\n" "$last_line" | awk '{print $NF}')

  # only include in average if size >= MIN_SIZE
  if (( sz >= MIN_SIZE )); then
    # numeric? accept ints/decimals; if it’s not numeric, skip
    if [[ "$val" =~ ^[0-9]*\.?[0-9]+$ ]]; then
      sum=$(awk -v a="$sum" -v b="$val" 'BEGIN{printf("%.10f", a+b)}')
      ((count++))
    else
      echo "   [warn] could not parse a numeric last field for size ${sz} — continuing"
      ((fail++))
    fi
  fi
done

echo
if (( count > 0 )); then
  avg=$(awk -v s="$sum" -v c="$count" 'BEGIN{printf("%.4f", s/c)}')
  echo "Successful runs: ${count}; Failed: ${fail}"
  echo "Average over ${count} runs: ${avg}"
else
  echo "No successful runs to average. Failed: ${fail}"
fi
