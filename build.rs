fn main() {
    // A Cargo version alone is not a persistence identity: two dirty source
    // trees can legitimately report the same package version.  Hash the
    // engine inputs deterministically without depending on a VCS checkout.
    let manifest = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut files = Vec::new();
    collect_rust_files(&manifest.join("src"), &mut files);
    files.push(manifest.join("Cargo.toml"));
    files.push(manifest.join("build.rs"));
    let lock = manifest.join("Cargo.lock");
    if lock.exists() {
        files.push(lock);
    }
    files.sort();
    let mut hash = 0xcbf29ce484222325u64;
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        hash_bytes(
            &mut hash,
            path.strip_prefix(&manifest)
                .unwrap()
                .to_string_lossy()
                .as_bytes(),
        );
        hash_bytes(&mut hash, &std::fs::read(path).unwrap());
    }
    let out =
        std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("snapshot_engine.rs");
    let target = std::env::var("TARGET").unwrap_or_default();
    let profile = std::env::var("PROFILE").unwrap_or_default();
    let rustc_output =
        std::process::Command::new(std::env::var_os("RUSTC").expect("Cargo provides RUSTC"))
            .arg("--version")
            .arg("--verbose")
            .output()
            .expect("read Cargo compiler identity");
    assert!(
        rustc_output.status.success(),
        "cannot identify Cargo compiler"
    );
    let rustc = String::from_utf8(rustc_output.stdout).expect("compiler identity is UTF-8");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    let rustflags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    std::fs::write(out, format!(
        "pub const SNAPSHOT_ENGINE_SOURCE_HASH: &str = \"{hash:016x}\";\npub const SNAPSHOT_ENGINE_TARGET: &str = {target:?};\npub const SNAPSHOT_ENGINE_PROFILE: &str = {profile:?};\npub const SNAPSHOT_ENGINE_RUSTC: &str = {rustc:?};\npub const SNAPSHOT_ENGINE_RUSTFLAGS: &str = {rustflags:?};\n"
    ))
    .unwrap();
    if std::env::var_os("CARGO_FEATURE_PRIMA").is_some() {
        println!("cargo:rerun-if-env-changed=PRIMA_DIR");
        if let Some(prima_dir) = std::env::var_os("PRIMA_DIR") {
            let lib_dir = std::path::PathBuf::from(prima_dir).join("lib");
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
        }
        println!("cargo:rustc-link-lib=primac");
    }
}

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}

fn collect_rust_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_rust_files(&path, out);
        } else if path.extension().and_then(|v| v.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}
