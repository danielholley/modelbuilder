//! TypeScript declarations for the web UI, generated from the Rust types so
//! they are never written twice. `web/src/api/types.ts` is this output;
//! `tests/types.rs` fails when it is stale and rewrites it with
//! `UPDATE_TYPES=1 cargo test -p mb-server --test types`.

use std::any::TypeId;
use std::collections::{BTreeMap, HashSet};

use ts_rs::{Config, TypeVisitor, TS};

struct Collect<'a> {
    cfg: &'a Config,
    seen: HashSet<TypeId>,
    decls: BTreeMap<String, String>,
}

impl TypeVisitor for Collect<'_> {
    fn visit<T: TS + 'static + ?Sized>(&mut self) {
        // Primitives and wrappers have no output path: they are inlined.
        if T::output_path().is_none() || !self.seen.insert(TypeId::of::<T>()) {
            return;
        }
        let mut decl = T::docs().unwrap_or_default();
        decl.push_str("export ");
        decl.push_str(&T::decl(self.cfg));
        self.decls.insert(T::ident(self.cfg), decl);
        T::visit_dependencies(self);
    }
}

/// Every type the API sends or accepts, with its dependencies, as one module.
pub fn typescript() -> String {
    // JSON numbers: u64 counts (parameters, bytes) stay well below 2^53.
    let cfg = Config::new().with_large_int("number");
    let mut c = Collect {
        cfg: &cfg,
        seen: HashSet::new(),
        decls: BTreeMap::new(),
    };
    c.visit::<crate::Health>();
    c.visit::<crate::ErrorBody>();
    c.visit::<mb_api::Catalog>();
    c.visit::<mb_api::InspectRequest>();
    c.visit::<mb_api::InspectResponse>();
    c.visit::<mb_api::StatsRequest>();
    c.visit::<mb_api::WeightStatsReport>();
    c.visit::<mb_api::PlanRequest>();
    c.visit::<mb_api::Plan>();
    c.visit::<mb_api::fs::ListRequest>();
    c.visit::<mb_api::fs::DirListing>();
    c.visit::<mb_api::jobs::JobSource>();
    c.visit::<mb_api::jobs::JobSummary>();
    c.visit::<mb_api::jobs::JobUpdate>();
    c.visit::<mb_api::jobs::Cursor>();

    let mut out = String::from(
        "// Generated from the Rust types by mb-server (src/types.rs). Do not edit:\n\
         // run `UPDATE_TYPES=1 cargo test -p mb-server --test types` after changing them.\n",
    );
    for decl in c.decls.values() {
        out.push('\n');
        out.push_str(decl.trim_end());
        out.push('\n');
    }
    out
}
