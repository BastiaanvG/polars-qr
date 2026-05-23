//! How a result is handed back to Polars.
//!
//! Every aggregate in this crate returns exactly one row: a struct whose fields carry the
//! numbers together with the labels needed to read them. Matrices are encoded as a list of
//! rows, so a caller can name both axes without a matrix type crossing into Python.

use faer::MatRef;
use polars::prelude::*;

/// One row holding the list `values`.
pub fn float_list(name: &str, values: &[f64]) -> Series {
    let inner = Series::new(name.into(), values);
    Series::new(name.into(), [inner])
}

/// One row holding the list `values`, or a null when there is nothing to report.
pub fn optional_float_list(name: &str, values: Option<&[f64]>) -> Series {
    match values {
        Some(values) => float_list(name, values),
        None => Series::full_null(name.into(), 1, &float_list_dtype()),
    }
}

/// One row holding the list `values`.
pub fn string_list(name: &str, values: &[String]) -> Series {
    let inner = Series::new(name.into(), values);
    Series::new(name.into(), [inner])
}

/// One row holding `matrix` as a list of its rows.
pub fn matrix_rows(name: &str, matrix: MatRef<'_, f64>) -> Series {
    let rows: Vec<Series> = (0..matrix.nrows())
        .map(|i| {
            let row: Vec<f64> = (0..matrix.ncols()).map(|j| matrix[(i, j)]).collect();
            Series::new(name.into(), row)
        })
        .collect();
    let inner = Series::new(name.into(), rows);
    Series::new(name.into(), [inner])
}

/// One row holding a single count.
pub fn count(name: &str, value: usize) -> Series {
    Series::new(name.into(), [value as u32])
}

/// One row holding a single number.
pub fn number(name: &str, value: f64) -> Series {
    Series::new(name.into(), [value])
}

/// One row holding a single string.
pub fn text(name: &str, value: &str) -> Series {
    Series::new(name.into(), [value])
}

/// Assemble `fields` into the single struct row that an aggregate returns.
pub fn struct_row(name: &str, fields: &[Series]) -> PolarsResult<Series> {
    Ok(StructChunked::from_series(name.into(), 1, fields.iter())?.into_series())
}

/// Assemble `fields` into the struct that a row-preserving operation returns.
pub fn struct_rows(name: &str, height: usize, fields: &[Series]) -> PolarsResult<Series> {
    Ok(StructChunked::from_series(name.into(), height, fields.iter())?.into_series())
}

/// Spread `values` back over the rows they were read from.
///
/// A row-preserving operation is handed the rows that survived the null policy, but has to
/// answer for every row it was given; the ones that were dropped come back as null.
pub fn scattered(name: &str, valid: &[bool], values: impl Iterator<Item = f64>) -> Series {
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new(name.into(), valid.len());
    let mut values = values;
    for kept in valid {
        if *kept {
            builder.append_value(values.next().unwrap_or(f64::NAN));
        } else {
            builder.append_null();
        }
    }
    builder.finish().into_series()
}

/// One row holding a blob.
pub fn binary_row(name: &str, bytes: Vec<u8>) -> Series {
    BinaryChunked::from_slice(name.into(), &[bytes.as_slice()]).into_series()
}

/// Stack one-row results into the column that a row-wise operation returns.
pub fn concatenate_rows(name: &str, rows: Vec<Series>, dtype: Field) -> PolarsResult<Series> {
    let Some((first, rest)) = rows.split_first() else {
        return Ok(Series::new_empty(name.into(), dtype.dtype()));
    };
    let mut stacked = first.clone();
    for row in rest {
        stacked.append(row)?;
    }
    stacked.rename(name.into());
    Ok(stacked)
}

/// The dtype of a field written by [`float_list`].
pub fn float_list_dtype() -> DataType {
    DataType::List(Box::new(DataType::Float64))
}

/// The dtype of a field written by [`string_list`].
pub fn string_list_dtype() -> DataType {
    DataType::List(Box::new(DataType::String))
}

/// The dtype of a field written by [`matrix_rows`].
pub fn matrix_dtype() -> DataType {
    DataType::List(Box::new(float_list_dtype()))
}

#[cfg(test)]
mod tests {
    use faer::Mat;

    use super::*;

    #[test]
    fn a_float_list_is_one_row() {
        let series = float_list("values", &[1.0, 2.0, 3.0]);

        assert_eq!(series.len(), 1);
        assert_eq!(series.dtype(), &float_list_dtype());
        assert_eq!(series.list().unwrap().get_as_series(0).unwrap().len(), 3);
    }

    #[test]
    fn a_matrix_is_encoded_row_by_row() {
        let matrix = Mat::from_fn(2, 3, |i, j| (3 * i + j) as f64);

        let series = matrix_rows("m", matrix.as_ref());

        assert_eq!(series.len(), 1);
        assert_eq!(series.dtype(), &matrix_dtype());
        let rows = series.list().unwrap().get_as_series(0).unwrap();
        assert_eq!(rows.len(), 2);
        let second = rows.list().unwrap().get_as_series(1).unwrap();
        assert_eq!(
            second
                .f64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<_>>(),
            [3.0, 4.0, 5.0]
        );
    }

    #[test]
    fn a_struct_row_keeps_its_field_order() {
        let fields = [count("n", 7), number("value", 1.5)];

        let series = struct_row("result", &fields).unwrap();

        assert_eq!(series.len(), 1);
        let dtype = series.dtype().clone();
        let DataType::Struct(fields) = dtype else {
            panic!("expected a struct")
        };
        assert_eq!(fields[0].name(), "n");
        assert_eq!(fields[1].name(), "value");
    }
}
