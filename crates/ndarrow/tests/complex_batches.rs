use approx::assert_abs_diff_eq;
use ndarray::{ArrayD, IxDyn, array, s};
use ndarrow::{
    array2_complex32_to_fixed_size_list, array2_complex64_to_fixed_size_list,
    arrayd_complex32_to_fixed_shape_tensor, arrayd_complex64_to_fixed_shape_tensor,
    complex32_as_array_view2, complex32_fixed_shape_tensor_as_array_viewd,
    complex64_as_array_view2, complex64_fixed_shape_tensor_as_array_viewd,
};
use num_complex::{Complex32, Complex64};

#[test]
fn complex32_matrix_roundtrip_from_offset_array() {
    let base = array![
        [Complex32::new(10.0_f32, 1.0), Complex32::new(20.0, 2.0), Complex32::new(30.0, 3.0),],
        [Complex32::new(40.0, 4.0), Complex32::new(50.0, 5.0), Complex32::new(60.0, 6.0),],
        [Complex32::new(70.0, 7.0), Complex32::new(80.0, 8.0), Complex32::new(90.0, 9.0),],
    ];
    let sliced = base.slice_move(s![1.., 1..]);
    let expected = sliced.iter().copied().collect::<Vec<_>>();

    let storage = array2_complex32_to_fixed_size_list(sliced)
        .expect("complex32 matrix outbound should succeed for sliced arrays");
    let view = complex32_as_array_view2(&storage).expect("complex32 matrix inbound should succeed");

    assert_eq!(view.dim(), (2, 2));
    for (actual, expected) in view.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual.re, expected.re);
        assert_abs_diff_eq!(actual.im, expected.im);
    }
}

#[test]
fn complex64_matrix_roundtrip_from_offset_array() {
    let base = array![
        [Complex64::new(10.0_f64, 1.0), Complex64::new(20.0, 2.0), Complex64::new(30.0, 3.0),],
        [Complex64::new(40.0, 4.0), Complex64::new(50.0, 5.0), Complex64::new(60.0, 6.0),],
        [Complex64::new(70.0, 7.0), Complex64::new(80.0, 8.0), Complex64::new(90.0, 9.0),],
    ];
    let sliced = base.slice_move(s![1.., 1..]);
    let expected = sliced.iter().copied().collect::<Vec<_>>();

    let storage = array2_complex64_to_fixed_size_list(sliced)
        .expect("complex64 matrix outbound should succeed for sliced arrays");
    let view = complex64_as_array_view2(&storage).expect("complex64 matrix inbound should succeed");

    assert_eq!(view.dim(), (2, 2));
    for (actual, expected) in view.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual.re, expected.re);
        assert_abs_diff_eq!(actual.im, expected.im);
    }
}

#[test]
fn complex32_fixed_shape_tensor_roundtrip_from_offset_tensor() {
    let base = ArrayD::from_shape_vec(
        IxDyn(&[3, 2, 2]),
        vec![
            Complex32::new(1.0_f32, 0.0),
            Complex32::new(2.0, 1.0),
            Complex32::new(3.0, -1.0),
            Complex32::new(4.0, 0.5),
            Complex32::new(5.0, -0.5),
            Complex32::new(6.0, 2.0),
            Complex32::new(7.0, 0.25),
            Complex32::new(8.0, -2.5),
            Complex32::new(9.0, 1.5),
            Complex32::new(10.0, -0.75),
            Complex32::new(11.0, 0.125),
            Complex32::new(12.0, -3.0),
        ],
    )
    .expect("shape must be valid");
    let sliced = base.slice_move(s![1.., .., ..]).into_dyn();
    let expected = sliced.iter().copied().collect::<Vec<_>>();

    let (field, storage) = arrayd_complex32_to_fixed_shape_tensor("tensor32", sliced)
        .expect("complex32 fixed-shape tensor outbound should succeed for sliced arrays");
    let view = complex32_fixed_shape_tensor_as_array_viewd(&field, &storage)
        .expect("complex32 fixed-shape tensor inbound should succeed");

    assert_eq!(view.shape(), &[2, 2, 2]);
    for (actual, expected) in view.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual.re, expected.re);
        assert_abs_diff_eq!(actual.im, expected.im);
    }
}

#[test]
fn complex64_fixed_shape_tensor_roundtrip_from_offset_tensor() {
    let base = ArrayD::from_shape_vec(
        IxDyn(&[3, 2, 2]),
        vec![
            Complex64::new(1.0_f64, 0.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(4.0, 0.5),
            Complex64::new(5.0, -0.5),
            Complex64::new(6.0, 2.0),
            Complex64::new(7.0, 0.25),
            Complex64::new(8.0, -2.5),
            Complex64::new(9.0, 1.5),
            Complex64::new(10.0, -0.75),
            Complex64::new(11.0, 0.125),
            Complex64::new(12.0, -3.0),
        ],
    )
    .expect("shape must be valid");
    let sliced = base.slice_move(s![1.., .., ..]).into_dyn();
    let expected = sliced.iter().copied().collect::<Vec<_>>();

    let (field, storage) = arrayd_complex64_to_fixed_shape_tensor("tensor64", sliced)
        .expect("complex64 fixed-shape tensor outbound should succeed for sliced arrays");
    let view = complex64_fixed_shape_tensor_as_array_viewd(&field, &storage)
        .expect("complex64 fixed-shape tensor inbound should succeed");

    assert_eq!(view.shape(), &[2, 2, 2]);
    for (actual, expected) in view.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual.re, expected.re);
        assert_abs_diff_eq!(actual.im, expected.im);
    }
}
