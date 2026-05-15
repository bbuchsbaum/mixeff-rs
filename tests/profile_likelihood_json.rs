use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{DataFrame, LinearMixedModel};
use mixeff_rs::stats::{
    profile_confint_payload, ProfileLikelihoodCiPayload, PROFILE_LIKELIHOOD_CI_SCHEMA,
    PROFILE_LIKELIHOOD_CI_SCHEMA_VERSION,
};

fn dyestuff_like_data() -> DataFrame {
    let mut yield_values = Vec::new();
    let mut batch = Vec::new();
    for group in 0..6 {
        let batch_shift = (group as f64 - 2.5) * 3.0;
        for replicate in 0..5 {
            let noise = ((group + replicate) % 3) as f64 - 1.0;
            yield_values.push(1500.0 + batch_shift + noise);
            batch.push(format!("B{group}"));
        }
    }
    let mut data = DataFrame::new();
    data.add_numeric("yield", yield_values).unwrap();
    data.add_categorical("batch", batch).unwrap();
    data
}

#[test]
fn profile_confint_payload_round_trips_as_json() {
    let data = dyestuff_like_data();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let payload = profile_confint_payload(&mut model, 0.95).unwrap();
    assert_eq!(payload.schema_name, PROFILE_LIKELIHOOD_CI_SCHEMA);
    assert_eq!(payload.schema_version, PROFILE_LIKELIHOOD_CI_SCHEMA_VERSION);
    assert_eq!(payload.fit_criterion, "ML");
    assert!(payload.intervals.iter().any(|row| row.parameter == "β1"));
    assert!(payload.intervals.iter().any(|row| row.parameter == "σ"));
    assert!(!payload.profile_rows.is_empty());

    let json = payload.to_json().unwrap();
    let decoded = ProfileLikelihoodCiPayload::from_json(&json).unwrap();
    assert_eq!(decoded.schema_name, payload.schema_name);
    assert_eq!(decoded.schema_version, payload.schema_version);
    assert_eq!(decoded.intervals.len(), payload.intervals.len());
    assert_eq!(decoded.profile_rows.len(), payload.profile_rows.len());
    for (actual, expected) in decoded.intervals.iter().zip(payload.intervals.iter()) {
        assert_eq!(actual.parameter, expected.parameter);
        assert!((actual.estimate - expected.estimate).abs() < 1e-10);
        assert!((actual.lower - expected.lower).abs() < 1e-10);
        assert!((actual.upper - expected.upper).abs() < 1e-10);
    }
}

#[test]
fn reml_profile_payload_explicitly_notes_beta_omission() {
    let data = dyestuff_like_data();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let payload = profile_confint_payload(&mut model, 0.90).unwrap();
    assert_eq!(payload.fit_criterion, "REML");
    assert!(!payload
        .intervals
        .iter()
        .any(|row| row.parameter.starts_with("β")));
    assert!(payload
        .notes
        .iter()
        .any(|note| note.contains("omit fixed-effect beta profiles")));
}
