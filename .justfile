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
    cargo bench -p ndarrow --bench inbound_benchmarks -- --quick
    cargo bench -p ndarrow --bench outbound_benchmarks -- --quick
    echo "Benchmark smoke tests complete."

bench-report:
    bash ./scripts/bench_report.sh

bench-report-check:
    BENCH_FAIL_ON_REGRESSION=1 bash ./scripts/bench_report.sh

bench-baseline-update:
    cargo bench -p ndarrow --bench inbound_benchmarks -- --save-baseline main
    cargo bench -p ndarrow --bench outbound_benchmarks -- --save-baseline main

bench-smoke-report:
    just -f {{ justfile() }} bench-smoke
    just -f {{ justfile() }} bench-report

bench-smoke-check:
    just -f {{ justfile() }} bench-smoke
    just -f {{ justfile() }} bench-report-check

# --- RELEASE ---

prepare-release version:
    #!/usr/bin/env bash
    set -euo pipefail
    # Validate version format
    if ! [[ "{{ version }}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Error: Version must be in format X.Y.Z (e.g., 0.2.0)"
        exit 1
    fi

    # Require clean tree for deterministic release prep.
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "Error: Working tree is not clean. Commit or stash changes first."
        exit 1
    fi

    # Require git-cliff and fail fast when unavailable.
    if ! command -v git-cliff >/dev/null 2>&1; then
        echo "Error: git-cliff is required for release preparation."
        echo "Install it with: cargo install git-cliff"
        exit 1
    fi

    # Create release branch
    git checkout -b "release-v{{ version }}"

    # Update version in root Cargo.toml (in [workspace.package] section)
    awk '/^\[workspace\.package\]/ {in_workspace_package=1} in_workspace_package && /^version = / {gsub(/"[^"]*"/, "\"{{ version }}\""); in_workspace_package=0} {print}' Cargo.toml > Cargo.toml.tmp && mv Cargo.toml.tmp Cargo.toml

    # Update ndarrow crate version references in README files (if they exist).
    # Look for patterns like: ndarrow = "0.1.0" or ndarrow = { version = "0.1.0" }.
    for readme in README.md; do
        if [ -f "$readme" ]; then
            # Update simple dependency format
            sed -i '' "s/ndarrow = \"[0-9]*\.[0-9]*\.[0-9]*\"/ndarrow = \"{{ version }}\"/" "$readme" || true
            # Update version field in dependency table format
            sed -i '' "s/ndarrow = { version = \"[0-9]*\.[0-9]*\.[0-9]*\"/ndarrow = { version = \"{{ version }}\"/" "$readme" || true
        fi
    done

    # Update Cargo.lock
    cargo update --workspace

    # Verify leaf crate package locally.
    cargo package --allow-dirty -p ndarrow

    # Generate full changelog
    echo "Generating changelog..."
    git cliff -o CHANGELOG.md

    # Generate release notes for this version
    echo "Generating release notes..."
    git cliff --unreleased --tag v{{ version }} --strip header -o RELEASE_NOTES.md

    # Run all checks before preparing commit
    just -f {{ justfile() }} checks

    # Stage all changes.
    # Cargo.lock may be ignored in some setups, so stage it only if tracked.
    git add Cargo.toml CHANGELOG.md RELEASE_NOTES.md
    if git ls-files --error-unmatch Cargo.lock >/dev/null 2>&1; then
        git add Cargo.lock
    fi
    # Also add README if it was modified
    git add README.md 2>/dev/null || true

    # Commit
    git commit -m "chore: prepare release v{{ version }}"

    # Push branch and set upstream so later `git push` works without extra flags.
    git push --set-upstream origin "release-v{{ version }}"

    echo ""
    echo "Release preparation complete!"
    echo ""
    echo "Release notes preview:"
    echo "----------------------"
    head -20 RELEASE_NOTES.md
    echo ""
    echo "Next steps:"
    echo "1. Create a PR from the 'release-v{{ version }}' branch"
    echo "2. Review and merge the PR"
    echo "3. After merge, run: just tag-release {{ version }}"
    echo ""

tag-release version:
    #!/usr/bin/env bash
    set -euo pipefail

    # Ensure we're on main and up to date
    git checkout main
    git pull origin main

    # Verify the version in Cargo.toml matches requested version
    CARGO_VERSION=$(awk '
      /^\[workspace\.package\]/ { in_workspace_package = 1; next }
      in_workspace_package && /^version = / {
        gsub(/"/, "", $3)
        print $3
        exit
      }
    ' Cargo.toml)

    if [ "${CARGO_VERSION}" != "{{ version }}" ]; then
        echo "Error: Cargo.toml version (${CARGO_VERSION}) does not match requested version ({{ version }})"
        echo "Did the release PR merge successfully?"
        exit 1
    fi

    # Verify publish path works.
    cargo publish --dry-run -p ndarrow --no-verify

    # Create and push tag
    git tag -a "v{{ version }}" -m "Release v{{ version }}"
    git push origin "v{{ version }}"

    echo ""
    echo "Tag v{{ version }} created and pushed!"
    echo "The release workflow will now run automatically."
    echo ""

release-dry version:
    @echo "This would:"
    @echo "1. Create branch: release-v{{ version }}"
    @echo "2. Update version to {{ version }} in:"
    @echo "   - Cargo.toml [workspace.package] version"
    @echo "   - README dependency snippets (if they contain ndarrow version references)"
    @echo "3. Run local package check for ndarrow (publish dry-run path)"
    @echo "4. Update Cargo.lock"
    @echo "5. Generate CHANGELOG.md"
    @echo "6. Generate RELEASE_NOTES.md"
    @echo "7. Run just checks"
    @echo "8. Create commit and push branch"
    @echo ""
    @echo "After PR merge, 'just tag-release {{ version }}' would:"
    @echo "1. Tag the merged commit as v{{ version }}"
    @echo "2. Verify publish dry-run path (ndarrow)"
    @echo "3. Push the tag (triggering release workflow)"

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
