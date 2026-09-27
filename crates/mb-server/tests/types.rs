//! `web/src/api/types.ts` must match the Rust types.

#[test]
fn web_types_are_up_to_date() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/src/api/types.ts");
    let fresh = mb_server::types::typescript();
    if std::env::var_os("UPDATE_TYPES").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &fresh).unwrap();
        return;
    }
    let current = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(
        current == fresh,
        "{} is stale: run `UPDATE_TYPES=1 cargo test -p mb-server --test types`",
        path.display()
    );
}
