//! The binary form a mergeable state travels in.
//!
//! A state is written by one machine and read by another, so the layout is spelled out
//! rather than derived: a fixed magic, a format version, and a hash of the schema the state
//! was built for. Reading checks all three before trusting a single number, which is what
//! keeps a state from being merged into one it has nothing to do with.

use polars::prelude::*;

/// Marks a blob as one of ours.
const MAGIC: [u8; 5] = *b"POLQR";

/// The version of the layout. A reader refuses anything it does not know.
pub const FORMAT_VERSION: u16 = 1;

/// Which kind of state a blob holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The augmented QR factor of a least-squares problem.
    LeastSquares = 1,
    /// The centred second moments of a set of columns.
    Covariance = 2,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::LeastSquares => "least-squares",
            Self::Covariance => "covariance",
        }
    }

    fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            1 => Some(Self::LeastSquares),
            2 => Some(Self::Covariance),
            _ => None,
        }
    }
}

/// A hash of everything two states must agree on before they can be merged.
///
/// It is written out with the state and compared on the way back in. The hash is FNV-1a,
/// spelled out here rather than taken from the standard library so that the same names give
/// the same hash on every machine and in every version of the compiler.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SchemaHash(u64);

impl SchemaHash {
    /// Start a hash of a state's schema.
    pub fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    /// Fold a byte into the hash.
    pub fn byte(mut self, value: u8) -> Self {
        self.0 ^= u64::from(value);
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        self
    }

    /// Fold a length-prefixed name into the hash.
    pub fn name(mut self, value: &str) -> Self {
        for byte in (value.len() as u64).to_le_bytes() {
            self = self.byte(byte);
        }
        for byte in value.as_bytes() {
            self = self.byte(*byte);
        }
        self
    }

    /// Fold a list of names into the hash, in order.
    pub fn names(mut self, values: &[String]) -> Self {
        self = self.byte(values.len() as u8);
        for value in values {
            self = self.name(value);
        }
        self
    }

    fn value(self) -> u64 {
        self.0
    }
}

/// Builds the bytes of a state.
pub struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    /// Start a state of `kind` whose schema hashes to `schema`.
    pub fn new(kind: Kind, schema: SchemaHash) -> Self {
        let mut bytes = Vec::with_capacity(64);
        bytes.extend_from_slice(&MAGIC);
        bytes.push(kind as u8);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&schema.value().to_le_bytes());
        Self { bytes }
    }

    /// Append a count.
    pub fn u32(&mut self, value: u32) -> &mut Self {
        self.bytes.extend_from_slice(&value.to_le_bytes());
        self
    }

    /// Append a count.
    pub fn u64(&mut self, value: u64) -> &mut Self {
        self.bytes.extend_from_slice(&value.to_le_bytes());
        self
    }

    /// Append a flag.
    pub fn flag(&mut self, value: bool) -> &mut Self {
        self.bytes.push(u8::from(value));
        self
    }

    /// Append a number.
    pub fn f64(&mut self, value: f64) -> &mut Self {
        self.bytes.extend_from_slice(&value.to_le_bytes());
        self
    }

    /// Append a run of numbers, without a length of its own.
    pub fn numbers(&mut self, values: impl IntoIterator<Item = f64>) -> &mut Self {
        for value in values {
            self.f64(value);
        }
        self
    }

    /// Append a list of names.
    pub fn names(&mut self, values: &[String]) -> &mut Self {
        self.u32(values.len() as u32);
        for value in values {
            self.u32(value.len() as u32);
            self.bytes.extend_from_slice(value.as_bytes());
        }
        self
    }

    /// The finished state.
    pub fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

/// Reads the bytes of a state, checking as it goes.
pub struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    schema: SchemaHash,
}

impl<'a> Reader<'a> {
    /// Open a state, checking that it is one of ours and of the kind that was expected.
    pub fn new(bytes: &'a [u8], expected: Kind) -> PolarsResult<Self> {
        if bytes.len() < 16 || bytes[..5] != MAGIC {
            polars_bail!(ComputeError: "this is not a polars-qr state");
        }
        let kind = Kind::from_byte(bytes[5])
            .ok_or_else(|| polars_err!(ComputeError: "unknown state kind {}", bytes[5]))?;
        if kind != expected {
            polars_bail!(
                ComputeError:
                "this is a {} state, but a {} state was expected",
                kind.name(), expected.name(),
            );
        }
        let version = u16::from_le_bytes([bytes[6], bytes[7]]);
        if version != FORMAT_VERSION {
            polars_bail!(
                ComputeError:
                "this {} state is version {}, and this build reads version {}",
                kind.name(), version, FORMAT_VERSION,
            );
        }
        let schema = SchemaHash(u64::from_le_bytes(
            bytes[8..16].try_into().expect("eight bytes"),
        ));
        Ok(Self {
            bytes,
            at: 16,
            schema,
        })
    }

    /// The hash of the schema this state was built for.
    pub fn schema(&self) -> SchemaHash {
        self.schema
    }

    fn take(&mut self, count: usize) -> PolarsResult<&'a [u8]> {
        let end = self.at + count;
        if end > self.bytes.len() {
            polars_bail!(ComputeError: "this state ends in the middle of a value");
        }
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }

    /// Read a count.
    pub fn u32(&mut self) -> PolarsResult<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }

    /// Read a count.
    pub fn u64(&mut self) -> PolarsResult<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }

    /// Read a flag.
    pub fn flag(&mut self) -> PolarsResult<bool> {
        Ok(self.take(1)?[0] != 0)
    }

    /// Read a number.
    pub fn f64(&mut self) -> PolarsResult<f64> {
        Ok(f64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }

    /// Read `count` numbers.
    pub fn numbers(&mut self, count: usize) -> PolarsResult<Vec<f64>> {
        (0..count).map(|_| self.f64()).collect()
    }

    /// Read a list of names.
    pub fn names(&mut self) -> PolarsResult<Vec<String>> {
        let count = self.u32()? as usize;
        (0..count)
            .map(|_| {
                let length = self.u32()? as usize;
                let bytes = self.take(length)?;
                String::from_utf8(bytes.to_vec())
                    .map_err(|_| polars_err!(ComputeError: "a name in this state is not text"))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> Vec<String> {
        vec!["alpha".to_string(), "beta".to_string()]
    }

    #[test]
    fn writes_what_it_reads() {
        let schema = SchemaHash::new().names(&names());
        let mut writer = Writer::new(Kind::Covariance, schema);
        writer.u32(2).u64(9).flag(true).numbers([1.5, -2.5]);
        writer.names(&names());

        let bytes = writer.finish();
        let mut reader = Reader::new(&bytes, Kind::Covariance).unwrap();

        assert_eq!(reader.u32().unwrap(), 2);
        assert_eq!(reader.u64().unwrap(), 9);
        assert!(reader.flag().unwrap());
        assert_eq!(reader.numbers(2).unwrap(), [1.5, -2.5]);
        assert_eq!(reader.names().unwrap(), names());
    }

    #[test]
    fn refuses_a_blob_that_is_not_a_state() {
        assert!(Reader::new(b"not a state at all", Kind::Covariance).is_err());
        assert!(Reader::new(b"", Kind::Covariance).is_err());
    }

    #[test]
    fn refuses_a_state_of_another_kind() {
        let bytes = Writer::new(Kind::LeastSquares, SchemaHash::new()).finish();

        assert!(Reader::new(&bytes, Kind::Covariance).is_err());
        assert!(Reader::new(&bytes, Kind::LeastSquares).is_ok());
    }

    #[test]
    fn refuses_a_version_it_does_not_know() {
        let mut bytes = Writer::new(Kind::Covariance, SchemaHash::new()).finish();
        bytes[6] = 99;

        assert!(Reader::new(&bytes, Kind::Covariance).is_err());
    }

    #[test]
    fn refuses_a_state_that_stops_early() {
        let mut writer = Writer::new(Kind::Covariance, SchemaHash::new());
        writer.f64(1.0);
        let bytes = writer.finish();

        let mut reader = Reader::new(&bytes[..bytes.len() - 2], Kind::Covariance).unwrap();
        assert!(reader.f64().is_err());
    }

    #[test]
    fn states_built_for_different_columns_do_not_agree() {
        let one = Writer::new(Kind::Covariance, SchemaHash::new().names(&names())).finish();
        let other = Writer::new(
            Kind::Covariance,
            SchemaHash::new().names(&["alpha".to_string(), "gamma".to_string()]),
        )
        .finish();

        let one = Reader::new(&one, Kind::Covariance).unwrap();
        let other = Reader::new(&other, Kind::Covariance).unwrap();
        assert_ne!(one.schema(), other.schema());
    }

    #[test]
    fn the_schema_hash_depends_on_the_order_of_the_names() {
        let forwards = SchemaHash::new().names(&names());
        let backwards = SchemaHash::new().names(&["beta".to_string(), "alpha".to_string()]);

        assert_ne!(forwards, backwards);
    }

    #[test]
    fn the_schema_hash_is_the_same_from_one_run_to_the_next() {
        // The value is written down so that a change to the hash cannot pass unnoticed:
        // states written by an older build would stop merging into newer ones.
        assert_eq!(
            SchemaHash::new().names(&names()).value(),
            0x405d_848e_1af6_35f2
        );
    }
}
