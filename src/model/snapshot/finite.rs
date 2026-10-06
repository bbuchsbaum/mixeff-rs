//! Reject nonfinite floats before serde_json can silently turn them into null.

use serde::ser::{self, Serialize};

pub(super) fn validate<T: Serialize + ?Sized>(value: &T) -> Result<(), serde_json::Error> {
    value.serialize(Finite)
}

struct Finite;

macro_rules! scalar {
    ($($method:ident($ty:ty)),* $(,)?) => {$(
        fn $method(self, _: $ty) -> Result<(), Self::Error> { Ok(()) }
    )*};
}

impl ser::Serializer for Finite {
    type Ok = ();
    type Error = serde_json::Error;
    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeTupleStruct = Self;
    type SerializeTupleVariant = Self;
    type SerializeMap = Self;
    type SerializeStruct = Self;
    type SerializeStructVariant = Self;

    scalar! { serialize_bool(bool), serialize_i8(i8), serialize_i16(i16),
    serialize_i32(i32), serialize_i64(i64), serialize_i128(i128),
    serialize_u8(u8), serialize_u16(u16), serialize_u32(u32),
    serialize_u64(u64), serialize_u128(u128), serialize_char(char),
    serialize_str(&str), serialize_bytes(&[u8]) }
    fn serialize_f32(self, value: f32) -> Result<(), Self::Error> {
        self.serialize_f64(f64::from(value))
    }
    fn serialize_f64(self, value: f64) -> Result<(), Self::Error> {
        if value.is_finite() {
            Ok(())
        } else {
            Err(ser::Error::custom(
                "snapshot contains a nonfinite float; JSON cannot preserve this state",
            ))
        }
    }
    fn serialize_none(self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<(), Self::Error> {
        validate(value)
    }
    fn serialize_unit(self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        validate(value)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        validate(value)
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Self, Self::Error> {
        Ok(self)
    }
    fn serialize_tuple(self, _: usize) -> Result<Self, Self::Error> {
        Ok(self)
    }
    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<Self, Self::Error> {
        Ok(self)
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self, Self::Error> {
        Ok(self)
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Self, Self::Error> {
        Ok(self)
    }
    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Self, Self::Error> {
        Ok(self)
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Self, Self::Error> {
        Ok(self)
    }
}

macro_rules! compound {
    ($trait:ident, $method:ident $(, $key:ident: $key_ty:ty)?) => {
        impl ser::$trait for Finite {
            type Ok = ();
            type Error = serde_json::Error;
            fn $method<T: Serialize + ?Sized>(&mut self, $($key: $key_ty,)? value: &T) -> Result<(), Self::Error> {
                $(let _ = $key;)?
                validate(value)
            }
            fn end(self) -> Result<(), Self::Error> { Ok(()) }
        }
    };
}
compound!(SerializeSeq, serialize_element);
compound!(SerializeTuple, serialize_element);
compound!(SerializeTupleStruct, serialize_field);
compound!(SerializeTupleVariant, serialize_field);
compound!(SerializeStruct, serialize_field, key: &'static str);
compound!(SerializeStructVariant, serialize_field, key: &'static str);
impl ser::SerializeMap for Finite {
    type Ok = ();
    type Error = serde_json::Error;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), Self::Error> {
        validate(key)
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
        validate(value)
    }
    fn end(self) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Serialize)]
    struct Certificate {
        optional_statistic: Option<f64>,
    }

    #[test]
    fn rejects_nonfinite_optional_evidence_before_json_discards_it() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let evidence = Certificate {
                optional_statistic: Some(value),
            };
            assert_eq!(
                serde_json::to_string(&evidence).unwrap(),
                "{\"optional_statistic\":null}"
            );
            assert!(validate(&evidence).is_err());
        }
        assert!(validate(&Certificate {
            optional_statistic: None
        })
        .is_ok());
        assert!(validate(&Certificate {
            optional_statistic: Some(0.12345678901234568)
        })
        .is_ok());
    }
}
