//! Plain-text rendering of an inspection report.

use std::fmt::Write;

use mb_analyze::Report;
use mb_ir::ModelIr;

pub fn count(n: u64) -> String {
    let n = n as f64;
    match n {
        n if n >= 1e12 => format!("{:.2}T", n / 1e12),
        n if n >= 1e9 => format!("{:.2}B", n / 1e9),
        n if n >= 1e6 => format!("{:.2}M", n / 1e6),
        n if n >= 1e3 => format!("{:.1}K", n / 1e3),
        n => format!("{n}"),
    }
}

pub fn bytes(b: f64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut v = b;
    let mut i = 0;
    while v >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{v:.0} B")
    } else {
        format!("{v:.2} {}", UNITS[i])
    }
}

fn opt<T: std::fmt::Display>(v: Option<T>) -> String {
    v.map_or_else(|| "?".into(), |v| v.to_string())
}

pub fn report(r: &Report) -> String {
    let mut s = String::new();
    let a = &r.architecture;
    let _ = writeln!(s, "{}", r.source.path);
    let _ = writeln!(
        s,
        "  format {:?}, {} file(s), {} tensors\n",
        r.source.format, r.source.files, r.source.tensors
    );

    let _ = writeln!(s, "ARCHITECTURE");
    let _ = writeln!(
        s,
        "  family        {} {}",
        opt(a.family.as_deref()),
        a.architectures.join(", ")
    );
    let _ = writeln!(s, "  layers        {}", a.num_layers);
    let _ = writeln!(s, "  pattern       {}", a.layer_pattern);
    let _ = writeln!(
        s,
        "  hidden        {}   vocab {}",
        opt(a.hidden_size),
        opt(a.vocab_size)
    );
    let _ = writeln!(s, "  context       {}", opt(a.max_positions));
    let _ = writeln!(
        s,
        "  rope          theta {}{}",
        opt(a.rope_theta),
        a.rope_scaling
            .as_ref()
            .map_or(String::new(), |v| format!(", scaling {v}"))
    );
    let _ = writeln!(s, "  tied embed    {}", opt(a.tie_word_embeddings));
    let _ = writeln!(s, "  MTP modules   {}", a.mtp_modules);
    let _ = writeln!(s, "  multimodal    {}\n", a.multimodal);

    let p = &r.params;
    let _ = writeln!(
        s,
        "PARAMETERS      {} total, {} active per token",
        count(p.total),
        count(p.active_per_token)
    );
    for (name, v) in [
        ("embedding", p.embedding),
        ("lm_head", p.lm_head),
        ("attention", p.attention),
        ("linear attn", p.linear_attention),
        ("dense ffn", p.ffn_dense),
        ("routed experts", p.moe_routed_experts),
        ("shared experts", p.moe_shared_experts),
        ("router", p.router),
        ("norms", p.norms),
        ("mtp", p.mtp),
        ("multimodal", p.multimodal),
        ("other", p.other),
    ] {
        if v > 0 {
            let pct = 100.0 * v as f64 / p.total.max(1) as f64;
            let _ = writeln!(s, "  {name:<15} {:>9}  {pct:5.1}%", count(v));
        }
    }

    let q = &r.quantization;
    let _ = writeln!(
        s,
        "\nQUANTIZATION    {} on disk, {:.2} bits/param overall, {:.2} in trunk layers",
        bytes(q.total_bytes as f64),
        q.bits_per_param,
        q.trunk_bits_per_param
    );
    for d in &q.by_dtype {
        let approx = if d.bytes_exact {
            ""
        } else {
            " (size inferred)"
        };
        let _ = writeln!(
            s,
            "  {:<14} {:>5} tensors {:>9} params {:>11}  {:.3} b/p{approx}",
            d.dtype,
            d.tensors,
            count(d.params),
            bytes(d.bytes as f64),
            d.bits_per_param
        );
    }
    if let Some(d) = &q.declared {
        let _ = writeln!(s, "  declared: {d}");
    }
    for n in &q.notes {
        let _ = writeln!(s, "  note: {n}");
    }

    let k = &r.kv_cache;
    let _ = writeln!(
        s,
        "\nKV CACHE        {} global, {} sliding-window, {} linear-attention layers",
        k.global_layers, k.windowed_layers, k.linear_layers
    );
    for pr in &k.precisions {
        let at = k.context.map_or(String::new(), |c| {
            format!(", {} at {} tokens", bytes(pr.bytes_at_context), c)
        });
        let _ = writeln!(
            s,
            "  {:<13} {:>11}/token{at}",
            pr.name,
            bytes(pr.bytes_per_token)
        );
    }
    if let Some(b) = k.linear_state_bytes {
        let _ = writeln!(
            s,
            "  linear state  {} per sequence (fixed)",
            bytes(b as f64)
        );
    }
    if !k.unknown_layers.is_empty() {
        let _ = writeln!(s, "  unknown dims for layers {:?}", k.unknown_layers);
    }
    for a in &k.assumptions {
        let _ = writeln!(s, "  assumes: {a}");
    }

    let pv = &r.provenance;
    let _ = writeln!(s, "\nPROVENANCE");
    let _ = writeln!(s, "  name          {}", opt(pv.name.as_deref()));
    let _ = writeln!(s, "  license       {}", opt(pv.license.as_deref()));
    if !pv.base_models.is_empty() {
        let _ = writeln!(s, "  base model    {}", pv.base_models.join(", "));
    }
    if let Some(d) = &pv.dtype {
        let _ = writeln!(s, "  dtype         {d}");
    }
    if let Some(c) = pv.original_context {
        let _ = writeln!(
            s,
            "  context       trained at {c}, configured for {}",
            opt(pv.configured_context)
        );
    }
    let _ = writeln!(s, "  chat template {}", pv.has_chat_template);
    let _ = writeln!(s, "  lineage       {}", pv.lineage_hint);

    if !r.warnings.is_empty() {
        let _ = writeln!(s, "\nWARNINGS");
        for w in &r.warnings {
            let _ = writeln!(s, "  {w}");
        }
    }
    s
}

pub fn tensors(ir: &ModelIr) -> String {
    let mut s = String::from("\nTENSORS\n");
    for (t, r) in ir.tensors() {
        let layer = r.layer.map_or("-".into(), |l| l.to_string());
        let _ = writeln!(
            s,
            "  {:<60} {:<12} {:<20} {:>10}  {:?}/{layer}/{:?}",
            t.name,
            t.dtype.to_string(),
            format!("{:?}", t.shape),
            bytes(t.n_bytes as f64),
            r.component,
            r.kind
        );
    }
    s
}
