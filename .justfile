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
    RUST_LOG={{ LOG }} cargo test -p ndarrow --tests -- --nocapture --show-output

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
    just -f {{ justfile() }} clippy-no-default
    just -f {{ justfile() }} clippy-all-features

clippy-no-default:
    cargo +stable clippy --workspace --all-targets --no-default-features -- -D warnings -W clippy::pedantic

clippy-all-features:
    cargo +stable clippy --workspace --all-targets --all-features -- -D warnings -W clippy::pedantic

clippy-fix:
    cargo clippy --workspace --all-targets --all-features --fix --allow-dirty

fix:
    cargo clippy --workspace --all-targets --all-features --fix --allow-dirty
    cargo +nightly fmt

# --- FEATURE CHECKS ---

check-features:
    cargo check --workspace --no-default-features
    cargo check --workspace --all-features

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
    cargo bench -p ndarrow --bench inbound_benchmarks -- --warm-up-time 1 --measurement-time 2 --sample-size 10
    cargo bench -p ndarrow --bench outbound_benchmarks -- --warm-up-time 1 --measurement-time 2 --sample-size 10
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
    if ! [[ "{{ version }}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Error: Version must be in format X.Y.Z (e.g., 0.2.0)"
        exit 1
    fi

    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "Error: Working tree is not clean. Commit or stash changes first."
        exit 1
    fi

    if ! command -v git-cliff >/dev/null 2>&1; then
        echo "Error: git-cliff is required for release preparation."
        echo "Install it with: cargo install git-cliff"
        exit 1
    fi

    VERSION="{{ version }}"
    BRANCH="release-v${VERSION}"

    git checkout -b "${BRANCH}"

    # Update [workspace.package] version in Cargo.toml.
    awk -v version="${VERSION}" '
      /^\[workspace\.package\]/ { in_workspace_package = 1 }
      in_workspace_package && /^version = / {
        sub(/"[^"]+"/, "\"" version "\"")
        in_workspace_package = 0
      }
      { print }
    ' Cargo.toml > Cargo.toml.tmp
    mv Cargo.toml.tmp Cargo.toml

    # Keep README dependency snippet in sync.
    perl -0pi.bak -e "s/ndarrow = \"[0-9]+\\.[0-9]+\\.[0-9]+\"/ndarrow = \"${VERSION}\"/g" README.md
    rm -f README.md.bak

    cargo update --workspace
    cargo package --allow-dirty -p ndarrow

    git-cliff -o CHANGELOG.md
    git-cliff --unreleased --tag "v${VERSION}" --strip header -o RELEASE_NOTES.md

    just -f {{ justfile() }} checks

    git add Cargo.toml Cargo.lock README.md RELEASE_NOTES.md
    if [ -f CHANGELOG.md ]; then
        git add CHANGELOG.md
    fi
    git commit -m "chore: prepare release v${VERSION}"
    git push --set-upstream origin "${BRANCH}"

    echo ""
    echo "Release preparation complete."
    echo "Next steps:"
    echo "1. Open and merge PR from ${BRANCH}"
    echo "2. Run: just tag-release ${VERSION}"

tag-release version:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! [[ "{{ version }}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Error: Version must be in format X.Y.Z (e.g., 0.2.0)"
        exit 1
    fi

    VERSION="{{ version }}"

    git checkout main
    git pull origin main

    CARGO_VERSION=$(awk '
      /^\[workspace\.package\]/ { in_workspace_package = 1; next }
      in_workspace_package && /^version = / {
        gsub(/"/, "", $3)
        print $3
        exit
      }
    ' Cargo.toml)

    if [[ "${CARGO_VERSION}" != "${VERSION}" ]]; then
        echo "Error: Cargo.toml workspace version (${CARGO_VERSION}) != requested version (${VERSION})"
        exit 1
    fi

    cargo publish --dry-run -p ndarrow --no-verify
    git tag -a "v${VERSION}" -m "Release v${VERSION}"
    git push origin "v${VERSION}"

    echo ""
    echo "Tag v${VERSION} pushed."
    echo "GitHub release workflow should start automatically."

release-dry version:
    @echo "This would:"
    @echo "1. Validate semver {{ version }}"
    @echo "2. Require a clean tree"
    @echo "3. Create and push branch release-v{{ version }}"
    @echo "4. Update Cargo.toml [workspace.package] version to {{ version }}"
    @echo "5. Update README dependency snippet"
    @echo "6. Update Cargo.lock and run cargo package -p ndarrow"
    @echo "7. Generate CHANGELOG.md and RELEASE_NOTES.md"
    @echo "8. Run just checks"
    @echo "9. Commit release prep"
    @echo ""
    @echo "After merge, tag-release would:"
    @echo "1. Verify main is up to date"
    @echo "2. Verify workspace version matches {{ version }}"
    @echo "3. Run cargo publish --dry-run -p ndarrow --no-verify"
    @echo "4. Create and push tag v{{ version }}"

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
