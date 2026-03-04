# Execution Tracker

## Purpose

This document tracks operational state: what is done, what is next, and what is needed.
Use this to resume work after context compaction.

## Resume Protocol

1. Read `docs/README.md` for orientation
2. Read `docs/DECISIONS.md` — never violate locked decisions
3. Read this file — find highest-priority open item
4. Read `docs/architecture.md` if the task touches structure
5. Start from the highest-priority open item below

---

## Done

| ID    | Description                                          | Phase |
|-------|------------------------------------------------------|-------|
| D-001 | Design conversation: dense type mappings finalized   | Pre   |
| D-002 | Design conversation: sparse CSR mapping identified   | Pre   |
| D-003 | Design conversation: tensor extension types chosen   | Pre   |
| D-004 | Design conversation: multi-vector strategy decided   | Pre   |
| D-005 | Design conversation: null handling three-tier design  | Pre   |
| D-006 | Design conversation: ownership transfer mechanism    | Pre   |
| D-007 | Design conversation: allocation contract defined     | Pre   |
| D-008 | Design conversation: trait hierarchy designed         | Pre   |
| D-009 | AGENTS.md created                                    | Pre   |
| D-010 | docs/ folder created with all planning documents     | Pre   |
| D-011 | DECISIONS.md with all locked decisions               | Pre   |
| D-012 | architecture.md with module tree and type mappings   | Pre   |
| D-013 | CAPABILITY_MATRIX.md with full scope inventory       | Pre   |
| D-014 | ROADMAP.md with phased implementation plan           | Pre   |
| D-015 | PERFORMANCE_CONTRACTS.md with allocation guarantees  | Pre   |
| D-016 | NABLED_CHANGES.md with upstream change tracker       | Pre   |
| D-017 | STATUS.md with current state snapshot                | Pre   |
| D-018 | .justfile with quality gate commands                 | Pre   |

---

## Next (Priority Order)

| ID    | Description                                          | Phase | Priority |
|-------|------------------------------------------------------|-------|----------|
| N-001 | Update Cargo.toml with arrow and ndarray deps        | 1     | P0       |
| N-002 | Implement NarrowError enum                           | 1     | P0       |
| N-003 | Implement NarrowElement trait + f32/f64 impls        | 1     | P0       |
| N-004 | Define AsNdarray and IntoArrow trait stubs           | 1     | P0       |
| N-005 | Create module tree (inbound/, outbound/, ext/, etc.) | 1     | P0       |
| N-006 | Wire lib.rs with modules and prelude                 | 1     | P0       |
| N-007 | Foundation phase tests                               | 1     | P0       |
| N-008 | PrimitiveArray -> ArrayView1 impl                    | 2     | P0       |
| N-009 | FixedSizeList -> ArrayView2 impl                     | 2     | P0       |
| N-010 | Array1 -> PrimitiveArray impl                        | 3     | P0       |
| N-011 | Array2 -> FixedSizeList impl                         | 3     | P0       |
| N-012 | Three-tier null API on all impls                     | 4     | P0       |
| N-013 | narrow.csr_matrix extension type definition          | 5     | P1       |
| N-014 | CsrView type definition                              | 5     | P1       |
| N-015 | Sparse inbound (extension type + two-column)         | 5     | P1       |
| N-016 | Sparse outbound                                       | 5     | P1       |
| N-017 | FixedShapeTensor inbound                              | 6     | P1       |
| N-018 | FixedShapeTensor outbound                             | 6     | P1       |
| N-019 | VariableShapeTensor inbound iterator                  | 7     | P1       |
| N-020 | VariableShapeTensor outbound                          | 7     | P1       |
| N-021 | Cast helper                                           | 8     | P2       |
| N-022 | Densify helper                                        | 8     | P2       |
| N-023 | Reshape helper                                        | 8     | P2       |
| N-024 | Property tests                                        | 9     | P2       |
| N-025 | Benchmarks                                            | 9     | P2       |
| N-026 | Allocation verification tests                         | 9     | P2       |
| N-027 | Documentation polish                                  | 10    | P2       |
| N-028 | CI pipeline                                           | 10    | P2       |
| N-029 | Publish preparation                                   | 10    | P2       |

---

## Needed (Unscheduled / Decisions Pending)

| ID    | Description                                          | Notes                              |
|-------|------------------------------------------------------|------------------------------------|
| K-001 | Determine exact arrow crate version to pin           | Check latest stable                |
| K-002 | Determine exact ndarray crate version to pin         | Match nabled's 0.17               |
| K-003 | Verify ScalarBuffer::from(Vec<T>) for f32/f64        | Confirm ArrowNativeType impls      |
| K-004 | Verify FixedShapeTensor API in arrow-rs              | May need canonical_extension_types feature |
| K-005 | Determine if arrow-rs VariableShapeTensor is stable  | Check feature gate status          |
| K-006 | Design CsrView ergonomics for nabled interop         | See NABLED_CHANGES.md              |
| K-007 | Decide on Complex32/Complex64 NarrowElement impls    | Arrow complex type support unclear |
| K-008 | Evaluate proptest vs quickcheck for property tests   | proptest preferred for shrinking   |
