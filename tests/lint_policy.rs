#[test]
fn numerical_range_loop_exceptions_must_not_be_crate_wide() {
    let crate_root = include_str!("../src/lib.rs");
    let mut remainder = crate_root;

    while let Some(start) = remainder.find("#![allow(") {
        let attribute = &remainder[start..];
        let end = attribute
            .find(")]")
            .expect("crate-level allow attribute must be terminated");
        let attribute = &attribute[..end + 2];
        assert!(
            !attribute.contains("clippy::needless_range_loop"),
            "numerical range-loop exceptions must be local and carry an algebra or accumulation-order reason"
        );
        remainder = &remainder[start + end + 2..];
    }
}
