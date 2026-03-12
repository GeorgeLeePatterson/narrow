use approx::assert_abs_diff_eq;
use arrow_array::types::Float64Type;
use ndarray::{ArrayD, IxDyn, s};
use ndarrow::{arrayd_to_fixed_shape_tensor, fixed_shape_tensor_as_array_viewd};

#[test]
fn fixed_shape_tensor_roundtrip_from_offset_tensor_f64() {
    let base = ArrayD::from_shape_vec(
        IxDyn(&[3, 2, 2]),
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    )
    .expect("shape must be valid");
    let sliced = base.slice_move(s![1.., .., ..]).into_dyn();
    let expected = sliced.iter().copied().collect::<Vec<_>>();

    let (field, storage) = arrayd_to_fixed_shape_tensor("tensor", sliced)
        .expect("fixed-shape tensor outbound should succeed for sliced arrays");
    let view = fixed_shape_tensor_as_array_viewd::<Float64Type>(&field, &storage)
        .expect("fixed-shape tensor inbound should succeed");

    assert_eq!(view.shape(), &[2, 2, 2]);
    for (actual, expected) in view.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(*actual, *expected);
    }
}
