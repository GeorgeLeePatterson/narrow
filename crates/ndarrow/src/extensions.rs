//! Extension type registration and deserialization utilities.
//!
//! Arrow Rust does not provide a global extension-type registry. This module
//! provides ndarrow's explicit registry of supported extension handlers.

use arrow_schema::{
    Field,
    extension::{ExtensionType, FixedShapeTensor, VariableShapeTensor},
};

use crate::{
    complex::{Complex32Extension, Complex64Extension},
    error::NdarrowError,
    sparse::CsrMatrixExtension,
    tensor::{parse_fixed_shape_extension, parse_variable_shape_extension},
};

/// A deserialized extension type supported by ndarrow.
#[derive(Debug, Clone, PartialEq)]
pub enum RegisteredExtension {
    /// `ndarrow.csr_matrix`
    CsrMatrix(CsrMatrixExtension),
    /// `arrow.fixed_shape_tensor`
    FixedShapeTensor(FixedShapeTensor),
    /// `arrow.variable_shape_tensor`
    VariableShapeTensor(VariableShapeTensor),
    /// `ndarrow.complex32`
    Complex32(Complex32Extension),
    /// `ndarrow.complex64`
    Complex64(Complex64Extension),
}

impl RegisteredExtension {
    /// Returns the extension type name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::CsrMatrix(_) => CsrMatrixExtension::NAME,
            Self::FixedShapeTensor(_) => FixedShapeTensor::NAME,
            Self::VariableShapeTensor(_) => VariableShapeTensor::NAME,
            Self::Complex32(_) => Complex32Extension::NAME,
            Self::Complex64(_) => Complex64Extension::NAME,
        }
    }
}

/// Returns the set of extension type names supported by ndarrow.
#[must_use]
pub fn registered_extension_names() -> &'static [&'static str] {
    &[
        CsrMatrixExtension::NAME,
        FixedShapeTensor::NAME,
        VariableShapeTensor::NAME,
        Complex32Extension::NAME,
        Complex64Extension::NAME,
    ]
}

/// Deserializes any supported extension type from a field.
///
/// This is ndarrow's explicit extension-handler registry for field-level
/// deserialization.
///
/// # Errors
///
/// Returns an error when:
/// - the field has no extension type name;
/// - the extension name is unknown to ndarrow;
/// - extension metadata/storage validation fails.
pub fn deserialize_registered_extension(
    field: &Field,
) -> Result<RegisteredExtension, NdarrowError> {
    let name = field.extension_type_name().ok_or_else(|| NdarrowError::InvalidMetadata {
        message: "field extension type name missing".to_owned(),
    })?;

    match name {
        CsrMatrixExtension::NAME => field
            .try_extension_type::<CsrMatrixExtension>()
            .map(RegisteredExtension::CsrMatrix)
            .map_err(NdarrowError::from),
        FixedShapeTensor::NAME => {
            parse_fixed_shape_extension(field).map(RegisteredExtension::FixedShapeTensor)
        }
        VariableShapeTensor::NAME => {
            parse_variable_shape_extension(field).map(RegisteredExtension::VariableShapeTensor)
        }
        Complex32Extension::NAME => field
            .try_extension_type::<Complex32Extension>()
            .map(RegisteredExtension::Complex32)
            .map_err(NdarrowError::from),
        Complex64Extension::NAME => field
            .try_extension_type::<Complex64Extension>()
            .map(RegisteredExtension::Complex64)
            .map_err(NdarrowError::from),
        other => Err(NdarrowError::InvalidMetadata {
            message: format!("unsupported extension type: {other}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use arrow_schema::DataType;
    use ndarray::{ArrayD, IxDyn};

    use super::*;
    use crate::{sparse::csr_to_extension_array, tensor::arrays_to_variable_shape_tensor};

    fn field_with_extension_name(name: &str, data_type: DataType) -> Field {
        let mut metadata = HashMap::new();
        metadata.insert("ARROW:extension:name".to_owned(), name.to_owned());
        Field::new("manual", data_type, false).with_metadata(metadata)
    }

    #[test]
    fn registered_extension_names_include_expected_entries() {
        let names = registered_extension_names();
        assert!(names.contains(&CsrMatrixExtension::NAME));
        assert!(names.contains(&FixedShapeTensor::NAME));
        assert!(names.contains(&VariableShapeTensor::NAME));
        assert!(names.contains(&Complex32Extension::NAME));
        assert!(names.contains(&Complex64Extension::NAME));
    }

    #[test]
    fn deserialize_registered_extension_rejects_missing_name() {
        let field = Field::new("no_ext", DataType::Float32, false);
        let err =
            deserialize_registered_extension(&field).expect_err("missing extension should fail");
        assert!(matches!(err, NdarrowError::InvalidMetadata { .. }));
    }

    #[test]
    fn deserialize_registered_extension_parses_complex32() {
        let data_type = DataType::new_fixed_size_list(DataType::Float32, 2, false);
        let mut field = Field::new("c32", data_type, false);
        field
            .try_with_extension_type(Complex32Extension)
            .expect("attaching complex32 extension should succeed");

        let extension = deserialize_registered_extension(&field)
            .expect("registered extension parsing should succeed");
        assert!(matches!(extension, RegisteredExtension::Complex32(_)));
        assert_eq!(extension.name(), Complex32Extension::NAME);
    }

    #[test]
    fn deserialize_registered_extension_parses_complex64() {
        let data_type = DataType::new_fixed_size_list(DataType::Float64, 2, false);
        let mut field = Field::new("c64", data_type, false);
        field
            .try_with_extension_type(Complex64Extension)
            .expect("attaching complex64 extension should succeed");

        let extension = deserialize_registered_extension(&field)
            .expect("registered extension parsing should succeed");
        assert!(matches!(extension, RegisteredExtension::Complex64(_)));
        assert_eq!(extension.name(), Complex64Extension::NAME);
    }

    #[test]
    fn deserialize_registered_extension_parses_fixed_shape_tensor() {
        let data_type = DataType::new_fixed_size_list(DataType::Float32, 2, false);
        let extension = FixedShapeTensor::try_new(DataType::Float32, vec![2], None, None)
            .expect("fixed-shape tensor extension creation should succeed");
        let field = Field::new("tensor", data_type, false).with_extension_type(extension);

        let parsed = deserialize_registered_extension(&field)
            .expect("registered extension parsing should succeed");
        assert!(matches!(parsed, RegisteredExtension::FixedShapeTensor(_)));
        assert_eq!(parsed.name(), FixedShapeTensor::NAME);
    }

    #[test]
    fn deserialize_registered_extension_parses_variable_shape_tensor() {
        let input = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0_f32, 2.0])
            .expect("array construction should succeed");
        let (field, _array) = arrays_to_variable_shape_tensor("ragged", vec![input], None)
            .expect("variable-shape tensor construction should succeed");

        let parsed = deserialize_registered_extension(&field)
            .expect("registered extension parsing should succeed");
        assert!(matches!(parsed, RegisteredExtension::VariableShapeTensor(_)));
        assert_eq!(parsed.name(), VariableShapeTensor::NAME);
    }

    #[test]
    fn deserialize_registered_extension_parses_csr_matrix() {
        let row_ptrs = vec![0_i32, 1_i32];
        let col_indices = vec![0_u32];
        let values = vec![1.0_f32];
        let (field, _array) = csr_to_extension_array("csr", 4, row_ptrs, col_indices, values)
            .expect("csr extension construction should succeed");

        let parsed = deserialize_registered_extension(&field)
            .expect("registered extension parsing should succeed");
        assert!(matches!(parsed, RegisteredExtension::CsrMatrix(_)));
        assert_eq!(parsed.name(), CsrMatrixExtension::NAME);
    }

    #[test]
    fn deserialize_registered_extension_rejects_unknown_name() {
        let field = field_with_extension_name("ndarrow.unknown", DataType::Float32);
        let err =
            deserialize_registered_extension(&field).expect_err("unknown extension name must fail");
        assert!(matches!(err, NdarrowError::InvalidMetadata { .. }));
    }
}
