use std::{collections::HashMap, sync::Arc};

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int32Array, ListArray, StructArray,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{
    DataType, Field,
    extension::{
        EXTENSION_TYPE_METADATA_KEY, EXTENSION_TYPE_NAME_KEY, ExtensionType, FixedShapeTensor,
        VariableShapeTensor,
    },
};
use ndarray::{ArrayD, IxDyn};
use ndarrow::{
    arrayd_to_fixed_shape_tensor, arrays_to_variable_shape_tensor, deserialize_registered_extension,
};

fn field_with_extension(
    data_type: DataType,
    extension_name: &str,
    metadata_json: Option<&str>,
) -> Field {
    let mut metadata = HashMap::new();
    metadata.insert(EXTENSION_TYPE_NAME_KEY.to_owned(), extension_name.to_owned());
    if let Some(metadata_json) = metadata_json {
        metadata.insert(EXTENSION_TYPE_METADATA_KEY.to_owned(), metadata_json.to_owned());
    }
    Field::new("field", data_type, false).with_metadata(metadata)
}

fn simple_list_f32_type() -> DataType {
    DataType::List(Arc::new(Field::new("item", DataType::Float32, false)))
}

fn simple_shape_type(dimensions: i32) -> DataType {
    DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, false)), dimensions)
}

#[test]
fn fixed_shape_extension_rejects_missing_metadata() {
    let tensor =
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0_f64, 2.0, 3.0, 4.0]).expect("shape valid");
    let (_field, storage) =
        arrayd_to_fixed_shape_tensor("tensor", tensor).expect("fixed-shape tensor should build");
    let field = field_with_extension(storage.data_type().clone(), FixedShapeTensor::NAME, None);

    let err = deserialize_registered_extension(&field).expect_err("missing metadata must fail");
    let message = err.to_string();
    assert!(message.contains("metadata missing"));
}

#[test]
fn fixed_shape_extension_rejects_non_fixed_size_list_storage() {
    let field =
        field_with_extension(DataType::Float64, FixedShapeTensor::NAME, Some(r#"{"shape":[2]}"#));

    let err = deserialize_registered_extension(&field).expect_err("wrong storage must fail");
    let message = err.to_string();
    assert!(message.contains("requires FixedSizeList storage"));
}

#[test]
fn variable_shape_extension_rejects_missing_metadata() {
    let arrays = vec![
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0_f32, 2.0]).expect("shape valid"),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0_f32, 4.0]).expect("shape valid"),
    ];
    let (_field, storage) = arrays_to_variable_shape_tensor("ragged", arrays, None)
        .expect("variable tensor should build");
    let field = field_with_extension(storage.data_type().clone(), VariableShapeTensor::NAME, None);

    let err = deserialize_registered_extension(&field).expect_err("missing metadata must fail");
    let message = err.to_string();
    assert!(message.contains("metadata missing"));
}

#[test]
fn variable_shape_extension_rejects_non_struct_storage() {
    let field = field_with_extension(DataType::Float32, VariableShapeTensor::NAME, Some(r"{}"));

    let err =
        deserialize_registered_extension(&field).expect_err("wrong top-level storage must fail");
    let message = err.to_string();
    assert!(message.contains("requires Struct storage"));
}

#[test]
fn variable_shape_extension_rejects_missing_data_field() {
    let field = field_with_extension(
        DataType::Struct(vec![Field::new("shape", simple_shape_type(1), false)].into()),
        VariableShapeTensor::NAME,
        Some(r"{}"),
    );

    let err = deserialize_registered_extension(&field).expect_err("missing data field must fail");
    let message = err.to_string();
    assert!(message.contains("missing 'data' field"));
}

#[test]
fn variable_shape_extension_rejects_missing_shape_field() {
    let field = field_with_extension(
        DataType::Struct(vec![Field::new("data", simple_list_f32_type(), false)].into()),
        VariableShapeTensor::NAME,
        Some(r"{}"),
    );

    let err = deserialize_registered_extension(&field).expect_err("missing shape field must fail");
    let message = err.to_string();
    assert!(message.contains("missing 'shape' field"));
}

#[test]
fn variable_shape_extension_rejects_wrong_data_field_type() {
    let field = field_with_extension(
        DataType::Struct(
            vec![
                Field::new("data", DataType::Float32, false),
                Field::new("shape", simple_shape_type(1), false),
            ]
            .into(),
        ),
        VariableShapeTensor::NAME,
        Some(r"{}"),
    );

    let err =
        deserialize_registered_extension(&field).expect_err("wrong data field type must fail");
    let message = err.to_string();
    assert!(message.contains("'data' field must be List"));
}

#[test]
fn variable_shape_extension_rejects_wrong_shape_field_type() {
    let field = field_with_extension(
        DataType::Struct(
            vec![
                Field::new("data", simple_list_f32_type(), false),
                Field::new(
                    "shape",
                    DataType::List(Arc::new(Field::new("item", DataType::Int32, false))),
                    false,
                ),
            ]
            .into(),
        ),
        VariableShapeTensor::NAME,
        Some(r"{}"),
    );

    let err =
        deserialize_registered_extension(&field).expect_err("wrong shape field type must fail");
    let message = err.to_string();
    assert!(message.contains("'shape' field must be FixedSizeList"));
}

#[test]
fn variable_shape_extension_rejects_invalid_metadata_json() {
    let field = field_with_extension(
        DataType::Struct(
            vec![
                Field::new("data", simple_list_f32_type(), false),
                Field::new("shape", simple_shape_type(1), false),
            ]
            .into(),
        ),
        VariableShapeTensor::NAME,
        Some("{invalid"),
    );

    let err = deserialize_registered_extension(&field).expect_err("invalid metadata must fail");
    let message = err.to_string();
    assert!(message.contains("metadata parse failed"));
}

#[test]
fn variable_shape_tensor_iter_rejects_field_storage_mismatch_after_parse() {
    let data_values = Float32Array::new(ScalarBuffer::from(vec![1.0_f32, 2.0]), None);
    let data_offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2_i32]));
    let data_array: ArrayRef = Arc::new(ListArray::new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        data_offsets,
        Arc::new(data_values),
        None,
    ));

    let shape_values = Int32Array::new(ScalarBuffer::from(vec![2_i32]), None);
    let shape_array: ArrayRef = Arc::new(FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Int32, false)),
        1,
        Arc::new(shape_values),
        None,
    ));

    let array = StructArray::new(
        vec![
            Field::new("data", data_array.data_type().clone(), false),
            Field::new("shape", shape_array.data_type().clone(), false),
        ]
        .into(),
        vec![data_array, shape_array],
        None,
    );

    let field = field_with_extension(
        DataType::Struct(
            vec![
                Field::new("data", simple_list_f32_type(), false),
                Field::new("shape", simple_shape_type(2), false),
            ]
            .into(),
        ),
        VariableShapeTensor::NAME,
        Some(r"{}"),
    );

    let result =
        ndarrow::variable_shape_tensor_iter::<arrow_array::types::Float32Type>(&field, &array);
    assert!(result.is_err(), "field/array storage mismatch must fail");
    let err = result.err().expect("field/array storage mismatch must fail");
    let message = err.to_string();
    assert!(message.contains("data type mismatch"));
}
