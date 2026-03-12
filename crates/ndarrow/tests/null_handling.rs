//! Integration tests for the three-tier null handling API.
//!
//! Null handling in ndarrow is explicit at the call site. These tests verify
//! all three tiers from the public API perspective.

use approx::assert_abs_diff_eq;
use arrow_array::Float64Array;
use ndarrow::{AsNdarray, NdarrowError};

// ─── Tier 1: Validated (as_ndarray) ───

#[test]
fn validated_succeeds_on_no_nulls() {
    let arr = Float64Array::from(vec![1.0, 2.0, 3.0]);
    let view = arr.as_ndarray().unwrap();
    assert_eq!(view.len(), 3);
    assert_abs_diff_eq!(view[0], 1.0);
    assert_abs_diff_eq!(view[2], 3.0);
}

#[test]
fn validated_fails_on_single_null() {
    let arr = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
    let err = arr.as_ndarray().unwrap_err();

    match err {
        NdarrowError::NullsPresent { null_count } => assert_eq!(null_count, 1),
        other => panic!("expected NullsPresent, got: {other}"),
    }
}

#[test]
fn validated_fails_on_all_nulls() {
    let arr = Float64Array::from(vec![None, None, None]);
    let err = arr.as_ndarray().unwrap_err();

    match err {
        NdarrowError::NullsPresent { null_count } => assert_eq!(null_count, 3),
        other => panic!("expected NullsPresent, got: {other}"),
    }
}

#[test]
fn validated_null_check_is_o1() {
    // Even with a massive array, null_count() is pre-computed.
    // This test just verifies it works on a large array.
    let data: Vec<f64> = (0..100_000).map(f64::from).collect();
    let arr = Float64Array::from(data);
    let view = arr.as_ndarray().unwrap();
    assert_eq!(view.len(), 100_000);
}

// ─── Tier 2: Unchecked (as_ndarray_unchecked) ───

#[test]
fn unchecked_succeeds_on_no_nulls() {
    let arr = Float64Array::from(vec![10.0, 20.0, 30.0]);
    let view = unsafe { arr.as_ndarray_unchecked() };
    assert_eq!(view.len(), 3);
    assert_abs_diff_eq!(view[1], 20.0);
}

// Note: calling as_ndarray_unchecked() on an array with nulls is UB.
// In debug builds it will panic via debug_assert. We don't test that here
// because triggering UB in tests is not useful — the contract is clear.

// ─── Tier 3: Masked (as_ndarray_masked) ───

#[test]
fn masked_no_nulls_returns_none_mask() {
    let arr = Float64Array::from(vec![1.0, 2.0, 3.0]);
    let (view, mask) = arr.as_ndarray_masked();

    assert_eq!(view.len(), 3);
    assert!(mask.is_none(), "no nulls should produce None mask");
}

#[test]
fn masked_with_nulls_returns_valid_mask() {
    let arr = Float64Array::from(vec![Some(1.0), None, Some(3.0), None, Some(5.0)]);
    let (view, mask) = arr.as_ndarray_masked();

    // View covers ALL positions — even null ones.
    assert_eq!(view.len(), 5);

    // Mask tells us which positions are valid.
    let nulls = mask.expect("should have a validity bitmap");
    assert!(nulls.is_valid(0));
    assert!(!nulls.is_valid(1));
    assert!(nulls.is_valid(2));
    assert!(!nulls.is_valid(3));
    assert!(nulls.is_valid(4));
}

#[test]
fn masked_view_can_be_used_with_mask() {
    // Demonstrate the intended usage pattern: use the mask to filter values.
    let arr = Float64Array::from(vec![Some(10.0), None, Some(30.0), None, Some(50.0)]);
    let (view, mask) = arr.as_ndarray_masked();

    let nulls = mask.unwrap();

    // Sum only non-null values using the mask.
    let sum: f64 =
        view.iter().enumerate().filter(|(i, _)| nulls.is_valid(*i)).map(|(_, v)| v).sum();

    assert_abs_diff_eq!(sum, 90.0); // 10 + 30 + 50
}

#[test]
fn masked_all_nulls() {
    let arr = Float64Array::from(vec![None, None, None]);
    let (view, mask) = arr.as_ndarray_masked();

    assert_eq!(view.len(), 3);
    let nulls = mask.unwrap();
    assert!(!nulls.is_valid(0));
    assert!(!nulls.is_valid(1));
    assert!(!nulls.is_valid(2));
}

// ─── FixedSizeList null handling ───

#[test]
fn fsl_validated_rejects_outer_nulls() {
    use std::sync::Arc;

    use arrow_array::FixedSizeListArray;
    use arrow_buffer::NullBuffer;
    use arrow_schema::Field;

    let values = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]);
    let field = Arc::new(Field::new("item", arrow_schema::DataType::Float64, false));
    let nulls = NullBuffer::from(vec![true, false]); // second row is null
    let fsl = FixedSizeListArray::new(field, 2, Arc::new(values), Some(nulls));

    let err =
        ndarrow::fixed_size_list_as_array2::<arrow_array::types::Float64Type>(&fsl).unwrap_err();

    match err {
        NdarrowError::NullsPresent { null_count } => assert_eq!(null_count, 1),
        other => panic!("expected NullsPresent, got: {other}"),
    }
}

#[test]
fn fsl_masked_returns_outer_bitmap() {
    use std::sync::Arc;

    use arrow_array::FixedSizeListArray;
    use arrow_buffer::NullBuffer;
    use arrow_schema::Field;

    let values = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let field = Arc::new(Field::new("item", arrow_schema::DataType::Float64, false));
    let nulls = NullBuffer::from(vec![true, false, true]); // row 1 is null
    let fsl = FixedSizeListArray::new(field, 2, Arc::new(values), Some(nulls));

    let (view, mask) =
        ndarrow::fixed_size_list_as_array2_masked::<arrow_array::types::Float64Type>(&fsl).unwrap();

    assert_eq!(view.dim(), (3, 2));
    let outer_nulls = mask.unwrap();
    assert!(outer_nulls.is_valid(0));
    assert!(!outer_nulls.is_valid(1));
    assert!(outer_nulls.is_valid(2));
}

#[test]
fn fsl_masked_rejects_inner_component_nulls() {
    use std::sync::Arc;

    use arrow_array::FixedSizeListArray;
    use arrow_schema::Field;

    let values = Float64Array::from(vec![Some(1.0), None, Some(3.0), Some(4.0)]);
    let field = Arc::new(Field::new("item", arrow_schema::DataType::Float64, true));
    let fsl = FixedSizeListArray::new(field, 2, Arc::new(values), None);

    let err = ndarrow::fixed_size_list_as_array2_masked::<arrow_array::types::Float64Type>(&fsl)
        .unwrap_err();

    match err {
        NdarrowError::NullsPresent { null_count } => assert_eq!(null_count, 1),
        other => panic!("expected NullsPresent, got: {other}"),
    }
}

// ─── Error display ───

#[test]
fn error_display_is_informative() {
    let arr = Float64Array::from(vec![Some(1.0), None]);
    let err = arr.as_ndarray().unwrap_err();
    let msg = err.to_string();

    assert!(msg.contains('1'), "should mention the null count");
    assert!(msg.contains("null"), "should mention nulls");
}
