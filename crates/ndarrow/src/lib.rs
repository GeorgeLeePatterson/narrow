//! # ndarrow
//!
//! Zero-copy bridge between [Apache Arrow](https://arrow.apache.org/) and
//! [ndarray](https://docs.rs/ndarray).
//!
//! ndarrow lets you move data between Arrow and ndarray with zero allocations
//! on the bridge path:
//!
//! - **Arrow → ndarray**: borrow Arrow buffers as ndarray views (O(1), no copy)
//! - **ndarray → Arrow**: transfer owned ndarray buffers into Arrow arrays (O(1), ownership move)
//!
//! # Quick Example
//!
//! ```
//! use arrow_array::Float64Array;
//! use ndarrow::{AsNdarray, IntoArrow};
//! use ndarray::Array1;
//!
//! // Arrow -> ndarray (zero-copy view)
//! let arrow_array = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]);
//! let view = arrow_array.as_ndarray().unwrap(); // ArrayView1<f64>, no allocation
//! assert_eq!(view[0], 1.0);
//!
//! // ndarray -> Arrow (zero-copy ownership transfer)
//! let result = Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]);
//! let arrow_result: Float64Array = result.into_arrow().unwrap();
//! assert_eq!(arrow_result.len(), 4);
//! ```
//!
//! # Null Handling
//!
//! Null semantics are explicit at the call site:
//!
//! ```
//! use arrow_array::Float64Array;
//! use ndarrow::AsNdarray;
//!
//! let array = Float64Array::from(vec![1.0, 2.0, 3.0]);
//!
//! // Validated: returns Err if nulls present (O(1) check)
//! let view = array.as_ndarray().unwrap();
//!
//! // Unchecked: caller guarantees no nulls (zero cost)
//! let view = unsafe { array.as_ndarray_unchecked() };
//!
//! // Masked: returns view + validity bitmap (zero allocation)
//! let (view, mask) = array.as_ndarray_masked();
//! ```
//!
//! # Performance Guarantee
//!
//! Bridge conversions are O(1) regardless of array size. The bridge creates views
//! (pointer + shape) or transfers buffer ownership. It never touches the data.

pub mod element;
pub mod error;
pub mod helpers;
pub mod inbound;
pub mod outbound;
pub mod sparse;
pub mod tensor;

// ─── Re-exports ───

pub use element::NdarrowElement;
pub use error::NdarrowError;
pub use inbound::AsNdarray;
// Re-export free functions for FixedSizeList conversions.
pub use inbound::{
    fixed_size_list_as_array2, fixed_size_list_as_array2_masked,
    fixed_size_list_as_array2_unchecked,
};
pub use outbound::IntoArrow;
pub use sparse::{
    CsrMatrixExtension, CsrMatrixMetadata, CsrView, csr_to_extension_array, csr_view_from_columns,
    csr_view_from_extension,
};
pub use tensor::{
    VariableShapeTensorIter, arrayd_to_fixed_shape_tensor, arrays_to_variable_shape_tensor,
    fixed_shape_tensor_as_array_viewd, variable_shape_tensor_iter,
};
