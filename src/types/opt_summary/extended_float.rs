//! Lossless encoding for optimizer-history objectives, where failed trial
//! evaluations and an unavailable initial objective legitimately use infinity.
//! Fitted parameters, final objectives and diagnostic evidence remain finite.

use serde::{Deserialize, Deserializer, Serializer};

pub(crate) fn serialize<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
    if value.is_finite() {
        serializer.serialize_f64(*value)
    } else {
        serializer.serialize_str(&format!("ieee754:{:016x}", value.to_bits()))
    }
}

pub(crate) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Stored {
        Number(f64),
        Nonfinite(String),
    }
    match Stored::deserialize(deserializer)? {
        Stored::Number(value) if value.is_finite() => Ok(value),
        Stored::Nonfinite(encoded) => {
            let bits = encoded
                .strip_prefix("ieee754:")
                .filter(|hex| hex.len() == 16)
                .and_then(|hex| u64::from_str_radix(hex, 16).ok());
            match bits.map(f64::from_bits) {
                Some(value) if !value.is_finite() => Ok(value),
                _ => Err(serde::de::Error::custom(
                    "invalid nonfinite optimizer-history float",
                )),
            }
        }
        _ => Err(serde::de::Error::custom("invalid optimizer-history float")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(serde::Serialize, Deserialize)]
    struct History(#[serde(with = "super")] f64);

    #[test]
    fn retains_trial_objective_bits_without_null_conversion() {
        for value in [
            0.12345678901234568,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0x7ff8000000000042),
        ] {
            let json = serde_json::to_string(&History(value)).unwrap();
            let restored: History = serde_json::from_str(&json).unwrap();
            assert_eq!(restored.0.to_bits(), value.to_bits());
        }
        assert!(serde_json::from_str::<History>("null").is_err());
    }
}
