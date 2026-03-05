#!/usr/bin/env bash
set -euo pipefail

criterion_dir="${1:-crates/ndarrow/target/criterion}"
report_dir="${BENCH_REPORT_DIR:-coverage/benchmarks}"
warn_pct="${BENCH_WARN_PCT:-5}"
fail_pct="${BENCH_FAIL_PCT:-10}"
min_baseline_ns="${BENCH_MIN_BASELINE_NS:-100}"
fail_on_regression="${BENCH_FAIL_ON_REGRESSION:-0}"

mkdir -p "${report_dir}"
summary_md="${report_dir}/summary.md"
summary_json="${report_dir}/summary.json"

if ! command -v jq >/dev/null 2>&1; then
    echo "Error: jq is required for benchmark reporting." >&2
    exit 1
fi

if [ ! -d "${criterion_dir}" ]; then
    cat > "${summary_md}" <<EOF
# Benchmark Summary

No criterion directory found at \`${criterion_dir}\`.
EOF
    cat > "${summary_json}" <<EOF
{"total":0,"checked":0,"warn":0,"fail":0,"improved":0,"skipped":0}
EOF
    cat "${summary_md}"
    exit 0
fi

change_files_list="$(mktemp)"
find "${criterion_dir}" -type f -path "*/change/estimates.json" | sort > "${change_files_list}"

if [ ! -s "${change_files_list}" ]; then
    cat > "${summary_md}" <<EOF
# Benchmark Summary

No baseline comparison files were found. Run benchmarks with an existing criterion cache first.
EOF
    cat > "${summary_json}" <<EOF
{"total":0,"checked":0,"warn":0,"fail":0,"improved":0,"skipped":0}
EOF
    cat "${summary_md}"
    rm -f "${change_files_list}"
    exit 0
fi

total=0
checked=0
warn=0
fail=0
improved=0
skipped=0

{
    echo "# Benchmark Summary"
    echo
    echo "| Benchmark | Base (ns) | New (ns) | Change | Status |"
    echo "|---|---:|---:|---:|---|"
} > "${summary_md}"

while IFS= read -r change_file; do
    total=$((total + 1))
    bench_root="${change_file%/change/estimates.json}"
    base_file="${bench_root}/base/estimates.json"
    new_file="${bench_root}/new/estimates.json"

    if [ ! -f "${base_file}" ] || [ ! -f "${new_file}" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    base_ns="$(jq -r '.mean.point_estimate // 0' "${base_file}")"
    new_ns="$(jq -r '.mean.point_estimate // 0' "${new_file}")"
    change_ratio="$(jq -r '.mean.point_estimate // 0' "${change_file}")"

    below_threshold="$(awk -v base="${base_ns}" -v min="${min_baseline_ns}" 'BEGIN { print (base < min) ? 1 : 0 }')"
    if [ "${below_threshold}" -eq 1 ]; then
        skipped=$((skipped + 1))
        continue
    fi

    checked=$((checked + 1))
    change_pct="$(awk -v c="${change_ratio}" 'BEGIN { printf "%.2f", c * 100 }')"
    status="ok"

    is_fail="$(awk -v c="${change_ratio}" -v t="${fail_pct}" 'BEGIN { print (c * 100 >= t) ? 1 : 0 }')"
    is_warn="$(awk -v c="${change_ratio}" -v t="${warn_pct}" 'BEGIN { print (c * 100 >= t) ? 1 : 0 }')"
    is_improved="$(awk -v c="${change_ratio}" -v t="${warn_pct}" 'BEGIN { print (c * 100 <= -t) ? 1 : 0 }')"

    if [ "${is_fail}" -eq 1 ]; then
        status="regression-fail"
        fail=$((fail + 1))
    elif [ "${is_warn}" -eq 1 ]; then
        status="regression-warn"
        warn=$((warn + 1))
    elif [ "${is_improved}" -eq 1 ]; then
        status="improved"
        improved=$((improved + 1))
    fi

    bench_name="${bench_root#${criterion_dir}/}"
    printf '| `%s` | %.2f | %.2f | %+0.2f%% | %s |\n' \
        "${bench_name}" "${base_ns}" "${new_ns}" "${change_pct}" "${status}" >> "${summary_md}"
done < "${change_files_list}"

rm -f "${change_files_list}"

cat >> "${summary_md}" <<EOF

- total comparisons discovered: ${total}
- comparisons evaluated (>= ${min_baseline_ns} ns baseline): ${checked}
- warnings (>= +${warn_pct}%): ${warn}
- failures (>= +${fail_pct}%): ${fail}
- improvements (<= -${warn_pct}%): ${improved}
- skipped (missing files or tiny baseline): ${skipped}
EOF

cat > "${summary_json}" <<EOF
{"total":${total},"checked":${checked},"warn":${warn},"fail":${fail},"improved":${improved},"skipped":${skipped}}
EOF

cat "${summary_md}"

if [ "${fail_on_regression}" -eq 1 ] && [ "${fail}" -gt 0 ]; then
    echo "Benchmark regression threshold exceeded (${fail} failure case(s))." >&2
    exit 1
fi
