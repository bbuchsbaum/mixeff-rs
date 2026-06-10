//! Temporary probe: native (TrustBQ) sleepstudy full-ML endpoint vs fixtures.
//! Requires `--no-default-features --features unstable-internals`.

#[cfg(feature = "unstable-internals")]
fn main() {
    use mixeff_rs::datasets;
    use mixeff_rs::formula::parse_formula;
    use mixeff_rs::model::LinearMixedModel;

    let (data, _) = datasets::load("sleepstudy").unwrap();
    let formula = parse_formula("Reaction ~ 1 + Days + (1 + Days | Subject)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();
    let optsum = model.optsum();
    println!("theta      = {:?}", model.theta());
    println!("sigma      = {:.12}", model.sigma());
    println!("fmin       = {:.12}", optsum.fmin);
    println!("feval      = {}", optsum.feval);
    println!("return     = {}", optsum.return_value);
    println!("optimizer  = {:?}", optsum.optimizer);
    println!("expected theta = [0.929190605786122, 0.0181657547680123, 0.222643205629365]");
    println!("expected fmin  = 1751.93934448899");
    println!("expected sigma = 25.5919070352053");
}

#[cfg(not(feature = "unstable-internals"))]
fn main() {
    eprintln!("probe_sleepstudy_native requires --features unstable-internals");
}
