# ndarrow — Documentation Index

## Direction Statement

ndarrow is a zero-copy bridge between Apache Arrow and ndarray. It enables Arrow-native systems to
leverage ndarray-native numerical computation without allocation overhead on the bridge path.

Design principles: algebraic, compositional, homomorphic, denotationally sound.
Vendor stance: agnostic. ndarrow knows Arrow and ndarray. Nothing else.

Current downstream-driven release checkpoint:

1. checkpoint 1 = `ndarrow` is ready for a new release under the full concept-first
   standalone / `rows-of-X` bridge contract

## Document Index (Reading Order)

| #  | Document                   | Purpose                                        |
|----|----------------------------|------------------------------------------------|
| 1  | `DECISIONS.md`             | Locked architectural decisions                 |
| 2  | `architecture.md`          | Module structure, trait hierarchy, type map     |
| 3  | `PERFORMANCE_CONTRACTS.md` | Allocation and copy guarantees                 |
| 4  | `CAPABILITY_MATRIX.md`     | Scope inventory with status                    |
| 5  | `ROADMAP.md`               | Phased implementation plan (the plan)          |
| 6  | `EXECUTION_TRACKER.md`     | Done / Next / Needed operational tracker       |
| 7  | `STATUS.md`                | Current state snapshot                         |
| 8  | `NABLED_CHANGES.md`        | Upstream changes needed and interim artifacts  |

## Context Resume Protocol

If context has been compacted or you are resuming work:

1. Read this file first
2. Read `DECISIONS.md` — never violate locked decisions
3. Read `EXECUTION_TRACKER.md` — find highest-priority open item
4. Read `architecture.md` if the task touches structure
5. Read `PERFORMANCE_CONTRACTS.md` if the task touches conversions
6. Start from the highest-priority open item in the tracker

## Context Sufficiency Check

After reading docs, verify you can answer all of the following:

1. What Arrow type represents a dense vector column? A sparse vector column? A tensor column?
2. What ndarray type does each Arrow type map to?
3. Which conversions are zero-copy and which allocate?
4. What traits define the element type bridge?
5. What are the three null handling tiers?
6. What canonical Arrow extension types does ndarrow use?
7. What custom extension types does ndarrow define?
8. What is the ownership transfer mechanism for ndarray -> Arrow?
9. What are the quality gates and coverage threshold?
10. What changes are pending in nabled, and what interim artifacts exist?

If you cannot answer all 10, re-read the relevant document.

## Scope Boundary

ndarrow bridges Arrow and ndarray. It does not:
- Implement numerical algorithms (that's nabled's job)
- Define query semantics (that's DataFusion's job)
- Know about vector stores (that's qdrant-datafusion's job)
- Perform serialization beyond Arrow IPC (that's Arrow's job)

ndarrow's only job is making the memory layout transition between these two array models
cost nothing.
