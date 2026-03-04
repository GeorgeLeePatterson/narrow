# AGENTS.md — narrow

## Mission

Build narrow into a zero-copy, zero-overhead bridge between Apache Arrow and ndarray. narrow is the
interoperability layer that allows Arrow-native systems (DataFusion, flight, IPC) to leverage
ndarray-native numerical libraries (nabled, and any future ndarray consumer) without sacrificing
performance.

narrow is **vendor-agnostic**. It knows nothing about Qdrant, DataFusion, or any specific producer
or consumer. It bridges two memory models — Arrow's columnar buffers and ndarray's strided arrays —
with zero allocations on the bridge path.

## Mandatory Context Bootstrap

Before making any changes, read the following documents in order:

1. `docs/README.md` — Navigation, direction, reading protocol
2. `docs/DECISIONS.md` — Locked architectural decisions (do not violate)
3. `docs/architecture.md` — Module structure, trait hierarchy, type mappings
4. `docs/PERFORMANCE_CONTRACTS.md` — Allocation and copy guarantees
5. `docs/CAPABILITY_MATRIX.md` — Scope and status of all capabilities
6. `docs/ROADMAP.md` — Phased implementation plan
7. `docs/EXECUTION_TRACKER.md` — Done / Next / Needed operational state
8. `docs/STATUS.md` — Current snapshot
9. `docs/NABLED_CHANGES.md` — Required upstream changes and interim artifacts

After reading, verify context sufficiency (see docs/README.md for checklist).

## Non-Negotiable Constraints

### 1. Zero-Copy Bridge

narrow never allocates memory on the bridge path. Converting Arrow to ndarray produces views
(pointer + shape + strides). Converting ndarray to Arrow transfers ownership of the underlying
buffer. The bridge is O(1) in both directions.

### 2. No Hidden Allocations

Every allocation point is explicit and visible at the call site. If a function allocates, it is
either:
- A helper clearly named for its purpose (e.g., `cast`, `densify`, `fill_nulls`)
- An `into_` method that transfers ownership (zero additional allocation)

Functions that produce views never allocate. This is enforced by return types (`ArrayView`, not
`Array`).

### 3. Algebraic and Compositional Design

All abstractions must be:
- **Algebraic**: Types form well-defined algebras. Operations compose and obey expected laws
  (associativity, identity, etc.).
- **Compositional**: Small primitives combine into larger operations. Column-level conversions
  compose into RecordBatch-level operations. The API is built from orthogonal, combinable pieces.
- **Homomorphic**: The bridge preserves structure. Arrow's FixedSizeList maps to ndarray's Array2.
  Arrow's FixedShapeTensor maps to ndarray's ArrayD. The shape, dtype, and contiguity invariants
  are preserved across the mapping.
- **Denotationally sound**: Every public function has a clear mathematical denotation. `as_array_view`
  denotes the identity on the underlying bytes (same data, different type interpretation).
  `into_arrow` denotes ownership transfer (same bytes, different ownership wrapper). There are no
  functions whose meaning is ambiguous.

### 4. Vendor Agnosticism

narrow has zero knowledge of any specific Arrow producer or consumer. It does not assume:
- A particular scalar type (f32 vs f64) — both are first-class
- A particular vector store (Qdrant, Milvus, Pinecone, etc.)
- A particular query engine (DataFusion, Polars, etc.)
- A particular embedding dimension or sparsity pattern

narrow bridges Arrow and ndarray. Nothing more, nothing less.

### 5. ndarray Independence

narrow uses ndarray as its numerical array substrate. It does not depend on nabled or any other
ndarray consumer. If nabled needs specific view types or index generics, those changes happen in
nabled, and narrow provides what ndarray provides. No nabled-specific types leak into narrow's
public API.

### 6. Proper Arrow Extension Types

Custom Arrow semantics use the `ExtensionType` trait from `arrow_schema::extension`. Canonical
Arrow extension types (e.g., `arrow.fixed_shape_tensor`, `arrow.variable_shape_tensor`) are used
where they exist. narrow defines its own extension types only where no canonical type exists
(e.g., `narrow.csr_matrix` for sparse). Extension types enable cross-language interop with
pyarrow, arrow-java, etc.

### 7. Explicit Null Handling

Null semantics are explicit at the call site:
- `as_array_view_unchecked` — caller guarantees no nulls, zero cost
- `as_array_view` — validates no nulls, returns `Result`, O(1)
- `as_array_view_masked` — returns view + optional validity bitmap, zero allocation

No implicit null handling. No silent NaN substitution. No hidden filtering.

### 8. Type Safety via Traits

The element type bridge is defined by traits (`NarrowElement` or equivalent) that connect Arrow's
`ArrowPrimitiveType` with ndarray's element requirements. Generic code is bounded by these traits.
Concrete implementations exist for f32, f64, and any additional types supported.

### 9. Unsafe is Acceptable, if Guarded

Zero-copy reinterpretation of memory buffers may require `unsafe`. This is acceptable when:
- The safety invariant is documented
- The invariant is validated by the surrounding safe code
- The `unsafe` block is minimal and auditable
- Tests exercise the boundary conditions

## Quality Gates

All of the following must pass before any PR is merged.

### Preferred: just

```bash
just checks                        # Runs all gates below in sequence
```

Individual gates:

```bash
just fmt                           # cargo +nightly fmt -- --check
just clippy                        # cargo clippy --workspace -- -D warnings
just check-features                # cargo check (no-default, all, default)
just test                          # cargo test --workspace (unit + integration)
just coverage-check                # cargo llvm-cov (threshold: 90%)
```

### Raw cargo (if just is unavailable)

```bash
cargo +nightly fmt -- --check
cargo clippy --workspace -- -D warnings
cargo check --workspace --no-default-features
cargo check --workspace --all-features
cargo test --workspace --lib -- --nocapture --show-output
cargo test -p narrow --tests -- --nocapture --show-output
```

### Coverage (if cargo-llvm-cov is available)

```bash
cargo llvm-cov clean --workspace
cargo llvm-cov --workspace --lib --tests --no-report
cargo llvm-cov report --fail-under-lines 90
```

### Coverage Requirement

Test line coverage must be >= 90%. No exceptions. If a code path exists, it has a test.

### Clippy Configuration

All warnings are errors (`-D warnings`). No `#[allow]` without a comment explaining why.

## Documentation Discipline

When making changes:

1. **New capability**: Add entry to `docs/CAPABILITY_MATRIX.md`
2. **New decision**: Add entry to `docs/DECISIONS.md` with rationale
3. **Completed work**: Update `docs/EXECUTION_TRACKER.md` (move to Done)
4. **Architecture change**: Update `docs/architecture.md`
5. **Performance-relevant change**: Verify against `docs/PERFORMANCE_CONTRACTS.md`
6. **Nabled dependency identified**: Add to `docs/NABLED_CHANGES.md`

Every public function has a doc comment that states:
- What it does (denotational: what mathematical operation it represents)
- Whether it allocates (and why, if so)
- Panics (if any, and under what conditions)
- Safety (if unsafe, what invariants must hold)

## Naming Conventions

- Crate: `narrow`
- Modules: `snake_case`
- Traits: `PascalCase` (e.g., `NarrowElement`, `AsNdarray`, `IntoArrow`)
- Functions: `snake_case`
- View-producing functions: `as_` prefix (e.g., `as_array_view`)
- Ownership-transferring functions: `into_` prefix (e.g., `into_arrow`)
- Fallible functions: return `Result<T, NarrowError>`
- Unchecked variants: `_unchecked` suffix

## Workspace Structure

narrow is a Cargo workspace. The module tree is organized by conversion direction
(inbound/outbound) and data category (dense/sparse/tensor).

```
narrow/                         # Workspace root
  Cargo.toml                    # Workspace manifest (deps, profiles, lints)
  crates/
    narrow/                     # Main library crate
      Cargo.toml
      src/
        lib.rs
        ...
```

## Dependencies

narrow depends on:
- `arrow` / `arrow-array` / `arrow-buffer` / `arrow-schema` (Apache Arrow for Rust)
- `ndarray` (N-dimensional arrays)

It does NOT depend on:
- `nabled` or any ndarray consumer
- `qdrant-client` or any vector store client
- `datafusion` or any query engine
- Any serialization framework beyond what Arrow provides
