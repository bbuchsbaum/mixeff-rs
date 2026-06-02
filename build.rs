fn main() {
    if std::env::var_os("CARGO_FEATURE_PRIMA").is_some() {
        println!("cargo:rerun-if-env-changed=PRIMA_DIR");
        if let Some(prima_dir) = std::env::var_os("PRIMA_DIR") {
            let lib_dir = std::path::PathBuf::from(prima_dir).join("lib");
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
        }
        println!("cargo:rustc-link-lib=primac");
    }
}
