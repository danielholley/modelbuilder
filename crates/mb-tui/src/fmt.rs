//! Number formatting, matching the CLI and the web UI.

pub fn count(n: u64) -> String {
    let units = ["", "K", "M", "B", "T"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1000.0 && i < units.len() - 1 {
        v /= 1000.0;
        i += 1;
    }
    match i {
        0 => n.to_string(),
        _ if v >= 100.0 => format!("{v:.0}{}", units[i]),
        _ if v >= 10.0 => format!("{v:.1}{}", units[i]),
        _ => format!("{v:.2}{}", units[i]),
    }
}

pub fn bytes(n: u64) -> String {
    let units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < units.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", units[i])
    }
}

pub fn pct(x: Option<f64>) -> String {
    x.filter(|v| v.is_finite())
        .map_or("–".into(), |v| format!("{:.1}%", 100.0 * v))
}

pub fn range(low: f64, high: f64) -> String {
    let f = |v: f64| {
        if v >= 100.0 {
            format!("{v:.0}")
        } else if v >= 10.0 {
            format!("{v:.1}")
        } else {
            format!("{v:.2}")
        }
    };
    if low == high {
        f(low)
    } else {
        format!("{}–{}", f(low), f(high))
    }
}

pub fn dtype(d: &mb_ir::DType) -> String {
    d.to_string()
}

pub fn effect(v: f64, unit: &str) -> String {
    match unit {
        "bytes" => bytes(v as u64),
        u if u.starts_with("params") => format!(
            "{}{}",
            count(v as u64),
            u.strip_prefix("params").unwrap_or("")
        ),
        u => format!(
            "{v}{}",
            if u.is_empty() {
                String::new()
            } else {
                format!(" {u}")
            }
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_like_the_other_uis() {
        assert_eq!(count(424_697_856), "425M");
        assert_eq!(count(27_300_000_000), "27.3B");
        assert_eq!(bytes(1536), "1.50 KiB");
        assert_eq!(pct(Some(0.844)), "84.4%");
        assert_eq!(range(3.2, 12.0), "3.20–12.0");
        assert_eq!(
            effect(126e6, "params (reuse layers)"),
            "126M (reuse layers)"
        );
    }
}
