fn main() {
    if std::env::var_os("CARGO_FEATURE_PRIMA").is_some() {
        println!("cargo:rustc-link-lib=primac");
    }
}
