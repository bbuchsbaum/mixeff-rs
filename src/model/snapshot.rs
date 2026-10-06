//! Versioned, engine-owned fitted-state persistence primitives.
//!
//! Payloads are deliberately opaque implementation records. They are
//! accepted only with matching engine source and build configuration, followed
//! by numerical reconstruction checks. This is not a portable interchange
//! format or an assertion of identical linked dependency/native binaries.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

mod finite;

use crate::compiler::CompilerPolicy;
use crate::error::{MixedModelError, Result};
use crate::formula::Formula;
use crate::model::data::{CategoricalContrast, DataFrame};

include!(concat!(env!("OUT_DIR"), "/snapshot_engine.rs"));

pub(crate) const SNAPSHOT_SCHEMA_VERSION: &str = "1.0.0";
pub(crate) const SNAPSHOT_SCHEMA_NAME: &str = "mixeff-rs.fitted-state";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TrainingRecipe {
    pub(crate) formula: Formula,
    pub(crate) frame: FrameSnapshot,
    pub(crate) compiler_policy: CompilerPolicy,
    pub(crate) weights: Option<Vec<f64>>,
}

impl TrainingRecipe {
    pub(crate) fn new(
        formula: Formula,
        frame: &DataFrame,
        compiler_policy: CompilerPolicy,
        weights: Option<Vec<f64>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            formula,
            frame: frame.snapshot_frame(),
            compiler_policy,
            weights,
        })
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let frame = DataFrame::from_snapshot_frame(&self.frame)?;
        if let Some(weights) = &self.weights {
            if weights.len() != frame.nrow() || weights.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(MixedModelError::InvalidArgument(
                    "snapshot weights have an invalid length or value".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct FrameSnapshot {
    pub(crate) columns: Vec<SnapshotColumn>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum SnapshotColumn {
    Numeric {
        name: String,
        values: Vec<f64>,
    },
    Categorical {
        name: String,
        values: Vec<String>,
        levels: Vec<String>,
        refs: Vec<u32>,
        contrast: Option<CategoricalContrast>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SnapshotEnvelope {
    schema_name: String,
    schema_version: String,
    crate_version: String,
    engine_source_hash: String,
    features: String,
    target: String,
    profile: String,
    rustc: String,
    rustflags: String,
    model_kind: String,
    payload_json: String,
    sha256: String,
}

pub(crate) fn feature_identity() -> String {
    let mut enabled = Vec::new();
    for (name, on) in [
        ("nlopt", cfg!(feature = "nlopt")),
        ("prima", cfg!(feature = "prima")),
        ("unstable-internals", cfg!(feature = "unstable-internals")),
        ("rayon", cfg!(feature = "rayon")),
        ("faer-backend", cfg!(feature = "faer-backend")),
    ] {
        if on {
            enabled.push(name);
        }
    }
    enabled.join(",")
}

pub(crate) fn encode_snapshot<T: Serialize>(payload: &T) -> Result<String> {
    finite::validate(payload).map_err(|error| MixedModelError::Unsupported(error.to_string()))?;
    let mut envelope = SnapshotEnvelope {
        schema_name: SNAPSHOT_SCHEMA_NAME.to_string(),
        schema_version: SNAPSHOT_SCHEMA_VERSION.to_string(),
        crate_version: env!("CARGO_PKG_VERSION").to_string(),
        engine_source_hash: SNAPSHOT_ENGINE_SOURCE_HASH.to_string(),
        features: feature_identity(),
        target: SNAPSHOT_ENGINE_TARGET.to_string(),
        profile: SNAPSHOT_ENGINE_PROFILE.to_string(),
        rustc: SNAPSHOT_ENGINE_RUSTC.to_string(),
        rustflags: SNAPSHOT_ENGINE_RUSTFLAGS.to_string(),
        model_kind: std::any::type_name::<T>().to_string(),
        payload_json: serde_json::to_string(payload).map_err(json_error)?,
        sha256: String::new(),
    };
    envelope.sha256 = digest(&envelope)?;
    serde_json::to_string(&envelope).map_err(json_error)
}

pub(crate) fn decode_snapshot<T: serde::de::DeserializeOwned>(json: &str) -> Result<T> {
    let envelope: SnapshotEnvelope = serde_json::from_str(json).map_err(json_error)?;
    if envelope.schema_name != SNAPSHOT_SCHEMA_NAME
        || envelope.schema_version != SNAPSHOT_SCHEMA_VERSION
    {
        return Err(MixedModelError::Unsupported(format!(
            "snapshot schema {} is incompatible with schema {}",
            envelope.schema_version, SNAPSHOT_SCHEMA_VERSION
        )));
    }
    if envelope.crate_version != env!("CARGO_PKG_VERSION")
        || envelope.engine_source_hash != SNAPSHOT_ENGINE_SOURCE_HASH
        || envelope.features != feature_identity()
        || envelope.target != SNAPSHOT_ENGINE_TARGET
        || envelope.profile != SNAPSHOT_ENGINE_PROFILE
        || envelope.rustc != SNAPSHOT_ENGINE_RUSTC
        || envelope.rustflags != SNAPSHOT_ENGINE_RUSTFLAGS
        || envelope.model_kind != std::any::type_name::<T>()
    {
        return Err(MixedModelError::Unsupported(
            "snapshot model kind, engine source, or build configuration is incompatible"
                .to_string(),
        ));
    }
    if envelope.sha256 != digest(&envelope)? {
        return Err(MixedModelError::InvalidArgument(
            "snapshot integrity checksum mismatch".to_string(),
        ));
    }
    serde_json::from_str(&envelope.payload_json).map_err(json_error)
}

fn json_error(error: serde_json::Error) -> MixedModelError {
    MixedModelError::InvalidArgument(format!("invalid fitted-state snapshot JSON: {error}"))
}

fn digest(envelope: &SnapshotEnvelope) -> Result<String> {
    // A JSON tuple supplies unambiguous framing. Hash the original payload
    // bytes, not a deserialized/normalized representation that may drop fields.
    let bytes = serde_json::to_vec(&(
        &envelope.schema_name,
        &envelope.schema_version,
        &envelope.crate_version,
        &envelope.engine_source_hash,
        &envelope.features,
        &envelope.target,
        &envelope.profile,
        &envelope.rustc,
        &envelope.rustflags,
        &envelope.model_kind,
        &envelope.payload_json,
    ))
    .map_err(json_error)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_integrity_covers_exact_payload_bytes_and_model_kind() {
        let json = encode_snapshot(&vec![0.12345678901234568_f64]).unwrap();
        assert_eq!(
            decode_snapshot::<Vec<f64>>(&json).unwrap(),
            vec![0.12345678901234568]
        );
        let mut envelope: SnapshotEnvelope = serde_json::from_str(&json).unwrap();
        envelope.payload_json = "[0.2]".into();
        assert!(decode_snapshot::<Vec<f64>>(&serde_json::to_string(&envelope).unwrap()).is_err());
        assert!(decode_snapshot::<Vec<String>>(&json).is_err());
        assert!(encode_snapshot(&Some(f64::NAN)).is_err());
    }

    #[test]
    fn snapshot_refuses_missing_and_incompatible_envelopes() {
        let json = encode_snapshot(&vec![1.0_f64]).unwrap();
        for key in [
            "schema_version",
            "crate_version",
            "engine_source_hash",
            "features",
        ] {
            let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
            value[key] = "incompatible".into();
            assert!(
                decode_snapshot::<Vec<f64>>(&value.to_string()).is_err(),
                "{key}"
            );
        }
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value.as_object_mut().unwrap().remove("payload_json");
        assert!(decode_snapshot::<Vec<f64>>(&value.to_string()).is_err());
    }
}

// Tests instrument the actual optimizer dispatches, rather than inferring
// non-search from an unchanged `feval` field.  Production calls compile away.
#[cfg(test)]
thread_local! { static OPTIMIZER_ENTRIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

#[cfg(test)]
pub(crate) fn record_optimizer_entry() {
    OPTIMIZER_ENTRIES.with(|count| count.set(count.get() + 1));
}
#[cfg(not(test))]
pub(crate) fn record_optimizer_entry() {}

#[cfg(test)]
pub(crate) struct OptimizerEntryGuard(usize);
#[cfg(test)]
impl OptimizerEntryGuard {
    pub(crate) fn expect_none() -> Self {
        Self(OPTIMIZER_ENTRIES.with(|count| count.get()))
    }
    pub(crate) fn entries_since(&self) -> usize {
        OPTIMIZER_ENTRIES.with(|count| count.get() - self.0)
    }
}
