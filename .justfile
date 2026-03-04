LOG := env('RUST_LOG', '')
coverage_line_threshold := "90"

default:
    @just --list

# --- TESTS ---

test:
    just -f {{ justfile() }} test-unit
    just -f {{ justfile() }} test-integration

test-unit:
    RUST_LOG={{ LOG }} cargo test --workspace --lib -- --nocapture --show-output

test-all-targets:
    RUST_LOG={{ LOG }} cargo test --workspace --all-targets -- --nocapture --show-output

test-one test_name:
    RUST_LOG={{ LOG }} cargo test --workspace "{{ test_name }}" -- --nocapture --show-output

test-integration:
    RUST_LOG={{ LOG }} cargo test -p narrow --tests -- --nocapture --show-output

test-integration-all:
    RUST_LOG={{ LOG }} cargo test --workspace --tests -- --nocapture --show-output

# --- DOCS ---

doc:
    cargo doc --workspace --no-deps --open

# --- COVERAGE ---

coverage:
    cargo llvm-cov clean --workspace
    cargo llvm-cov --workspace --lib --tests --no-report
    cargo llvm-cov report --html --output-dir coverage --open

coverage-json:
    cargo llvm-cov clean --workspace
    cargo llvm-cov --workspace --lib --tests --no-report
    cargo llvm-cov report --json --output-path coverage/cov.json

coverage-lcov:
    cargo llvm-cov clean --workspace
    cargo llvm-cov --workspace --lib --tests --no-report
    cargo llvm-cov report --lcov --output-path coverage/lcov.info

coverage-check:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo llvm-cov clean --workspace
    cargo llvm-cov --workspace --lib --tests --no-report
    COVERAGE=$(cargo llvm-cov report --json 2>/dev/null | jq -r '.data[0].totals.lines.percent')
    echo "Line coverage: ${COVERAGE}%"
    THRESHOLD={{ coverage_line_threshold }}
    if (( $(echo "$COVERAGE < $THRESHOLD" | bc -l) )); then
        echo "FAIL: Coverage ${COVERAGE}% is below threshold ${THRESHOLD}%"
        exit 1
    else
        echo "PASS: Coverage ${COVERAGE}% meets threshold ${THRESHOLD}%"
    fi

# --- CLIPPY AND FORMATTING ---

fmt:
    cargo +nightly fmt -- --check

fmt-fix:
    cargo +nightly fmt

clippy:
    cargo clippy --workspace -- -D warnings

clippy-fix:
    cargo clippy --workspace --fix --allow-dirty

fix:
    cargo clippy --workspace --fix --allow-dirty
    cargo +nightly fmt

# --- FEATURE CHECKS ---

check-features:
    cargo check --workspace --no-default-features
    cargo check --workspace --all-features
    cargo check --workspace

# --- MAINTENANCE ---

checks:
    just -f {{ justfile() }} fmt
    just -f {{ justfile() }} clippy
    just -f {{ justfile() }} check-features
    just -f {{ justfile() }} test
    just -f {{ justfile() }} coverage-check
    @echo ""
    @echo "All checks passed."

# --- BENCHMARKS ---

bench:
    cargo bench --workspace

bench-one bench_name:
    cargo bench --workspace -- "{{ bench_name }}"

bench-smoke:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running benchmark smoke tests..."
    cargo bench --workspace -- --warm-up-time 1 --measurement-time 2 --sample-size 10
    echo "Benchmark smoke tests complete."

bench-report:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running benchmarks with baseline comparison..."
    cargo bench --workspace -- --save-baseline current
    echo "Benchmarks complete. Results in target/criterion/"

bench-baseline-update:
    cargo bench --workspace -- --save-baseline main

# --- RELEASE ---

prepare-release version:
    #!/usr/bin/env bash
    set -euo pipefail
    VERSION="{{ version }}"
    echo "Preparing release v${VERSION}..."

    # Update workspace version
    sed -i '' "s/^version = \".*\"/version = \"${VERSION}\"/" Cargo.toml

    # Generate changelog
    if command -v git-cliff &>/dev/null; then
        git-cliff --tag "v${VERSION}" -o CHANGELOG.md
        echo "Changelog updated."
    else
        echo "git-cliff not found, skipping changelog generation."
    fi

    # Run all checks
    just -f {{ justfile() }} checks

    echo ""
    echo "Release v${VERSION} prepared."
    echo "Review changes, then commit and tag."

tag-release version:
    #!/usr/bin/env bash
    set -euo pipefail
    VERSION="{{ version }}"
    git tag -a "v${VERSION}" -m "Release v${VERSION}"
    echo "Tagged v${VERSION}. Push with: git push origin v${VERSION}"

release-dry:
    cargo package -p narrow --list
    cargo package -p narrow

# --- DEVELOPMENT SETUP ---

init-dev:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Installing development tools..."
    rustup component add clippy
    rustup toolchain install nightly --component rustfmt
    cargo install cargo-llvm-cov || true
    cargo install just || true
    cargo install git-cliff || true
    cargo install cargo-audit || true
    echo "Development tools installed."

check-outdated:
    cargo install cargo-outdated 2>/dev/null || true
    cargo outdated --workspace

audit:
    cargo audit
