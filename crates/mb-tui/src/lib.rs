//! Terminal dashboard: the same views as the web UI, for SSH and cloud boxes.
//!
//! [`App`] holds all state and is drawn by [`App::draw`]; [`run`] drives it in
//! a real terminal. Everything it shows comes from `mb-api`, so the web and
//! terminal dashboards can't disagree.

mod fmt;

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use mb_api::jobs::{Cursor, JobId, JobManager, JobSource, JobStatus, JobSummary};
use mb_api::{InspectRequest, InspectResponse, Plan, PlanRequest};
use mb_features::estimate::Fit;
use mb_ir::Mixer;
use mb_jobs::{Event, EventRecord};
use ratatui::crossterm::event::{
    self, Event as TermEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers,
};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Axis, Block, Borders, Cell, Chart, Dataset, Gauge, GraphType, Paragraph, Row, Table,
    TableState, Tabs, Wrap,
};
use ratatui::{DefaultTerminal, Frame};

use fmt::{bytes, count, pct, range};

// Categorical slots 1–2 and status colors, stepped for dark terminals.
const SERIES_1: Color = Color::Rgb(0x39, 0x87, 0xe5);
const SERIES_2: Color = Color::Rgb(0xd9, 0x59, 0x26);
const MUTED: Color = Color::Rgb(0x89, 0x87, 0x81);
const GOOD: Color = Color::Rgb(0x0c, 0xa3, 0x0c);
const WARNING: Color = Color::Rgb(0xfa, 0xb2, 0x19);
const CRITICAL: Color = Color::Rgb(0xd0, 0x3b, 0x3b);

#[derive(Clone, Debug, Default)]
pub struct Options {
    /// A .gguf file or HF directory to inspect and plan.
    pub model: Option<PathBuf>,
    /// A job to run (spec) or follow (events file).
    pub job: Option<JobSource>,
    /// Hardware profiles for the plan (all when empty).
    pub hardware: Vec<String>,
    /// Python interpreter for spec jobs.
    pub python: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tab {
    Overview,
    Tensors,
    Plan,
    Job,
}

const TABS: [(Tab, &str); 4] = [
    (Tab::Overview, "1 Overview"),
    (Tab::Tensors, "2 Tensors"),
    (Tab::Plan, "3 Plan"),
    (Tab::Job, "4 Job"),
];

struct JobState {
    id: JobId,
    summary: JobSummary,
    events: Vec<EventRecord>,
    log: Vec<String>,
    cursor: Cursor,
}

pub struct App {
    pub tab: Tab,
    model: Option<PathBuf>,
    inspect: Result<Option<InspectResponse>, String>,
    plan: Option<Result<Plan, String>>,
    plan_rx: Option<Receiver<Result<Plan, String>>>,
    jobs: JobManager,
    job: Option<JobState>,
    job_error: Option<String>,
    tensors: TableState,
    filter: String,
    filtering: bool,
    plan_scroll: u16,
    pub quit: bool,
}

impl App {
    /// Reads the model's headers now; the plan is computed on a background thread.
    pub fn new(opts: Options) -> Self {
        let inspect = match &opts.model {
            None => Ok(None),
            Some(p) => mb_api::inspect(&InspectRequest {
                path: p.display().to_string(),
                context: None,
                tensors: true,
            })
            .map(Some)
            .map_err(|e| e.to_string()),
        };
        let plan_rx = opts.model.as_ref().filter(|_| inspect.is_ok()).map(|p| {
            let (tx, rx) = mpsc::channel();
            let req = PlanRequest {
                path: Some(p.display().to_string()),
                hardware: opts.hardware.clone(),
                ..Default::default()
            };
            std::thread::spawn(move || {
                let _ = tx.send(mb_api::plan(&req).map_err(|e| e.to_string()));
            });
            rx
        });
        let jobs = JobManager::new(opts.python.clone());
        let (job, job_error) = match opts.job.clone().map(|src| jobs.start(src)) {
            None => (None, None),
            Some(Ok(summary)) => (
                Some(JobState {
                    id: summary.id,
                    summary,
                    events: Vec::new(),
                    log: Vec::new(),
                    cursor: Cursor::default(),
                }),
                None,
            ),
            Some(Err(e)) => (None, Some(e.to_string())),
        };
        let tab = if opts.model.is_none() && (job.is_some() || job_error.is_some()) {
            Tab::Job
        } else {
            Tab::Overview
        };
        let mut tensors = TableState::default();
        tensors.select(Some(0));
        Self {
            tab,
            model: opts.model,
            inspect,
            plan: None,
            plan_rx,
            jobs,
            job,
            job_error,
            tensors,
            filter: String::new(),
            filtering: false,
            plan_scroll: 0,
            quit: false,
        }
    }

    /// Picks up background results: the plan and new job events.
    pub fn tick(&mut self) {
        if let Some(rx) = &self.plan_rx {
            if let Ok(p) = rx.try_recv() {
                self.plan = Some(p);
                self.plan_rx = None;
            }
        }
        if let Some(j) = &mut self.job {
            if let Ok(u) = self.jobs.update(j.id, j.cursor) {
                j.events.extend(u.events);
                j.log.extend(u.log);
                j.summary = u.summary;
                j.cursor = u.cursor;
            }
        }
    }

    /// Whether background work is still pending (for tests).
    pub fn busy(&self) -> bool {
        self.plan_rx.is_some()
            || self
                .job
                .as_ref()
                .is_some_and(|j| j.summary.status == JobStatus::Running)
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        if key.kind != KeyEventKind::Press {
            return;
        }
        if self.filtering {
            match key.code {
                KeyCode::Esc => {
                    self.filtering = false;
                    self.filter.clear();
                }
                KeyCode::Enter => self.filtering = false,
                KeyCode::Backspace => {
                    self.filter.pop();
                }
                KeyCode::Char(c) => self.filter.push(c),
                _ => {}
            }
            self.tensors.select(Some(0));
            return;
        }
        let page = 10;
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => self.quit = true,
            KeyCode::Char('1') => self.tab = Tab::Overview,
            KeyCode::Char('2') => self.tab = Tab::Tensors,
            KeyCode::Char('3') => self.tab = Tab::Plan,
            KeyCode::Char('4') => self.tab = Tab::Job,
            KeyCode::Tab | KeyCode::Right => self.tab = next_tab(self.tab, 1),
            KeyCode::BackTab | KeyCode::Left => self.tab = next_tab(self.tab, 3),
            KeyCode::Char('/') if self.tab == Tab::Tensors => self.filtering = true,
            KeyCode::Char('c') if self.tab == Tab::Job => {
                if let Some(j) = &self.job {
                    let _ = self.jobs.cancel(j.id);
                }
            }
            KeyCode::Down | KeyCode::Char('j') => self.scroll(1),
            KeyCode::Up | KeyCode::Char('k') => self.scroll(-1),
            KeyCode::PageDown => self.scroll(page),
            KeyCode::PageUp => self.scroll(-page),
            KeyCode::Home | KeyCode::Char('g') => self.scroll(-100_000),
            _ => {}
        }
    }

    fn scroll(&mut self, by: i32) {
        match self.tab {
            Tab::Tensors => {
                let n = self.tensor_rows().len();
                let cur = self.tensors.selected().unwrap_or(0) as i32;
                self.tensors.select(Some(
                    (cur + by).clamp(0, n.saturating_sub(1) as i32) as usize
                ));
            }
            Tab::Plan => self.plan_scroll = (self.plan_scroll as i32 + by).max(0) as u16,
            _ => {}
        }
    }

    fn tensor_rows(&self) -> Vec<&mb_api::TensorRow> {
        let f = self.filter.to_lowercase();
        match &self.inspect {
            Ok(Some(r)) => r
                .tensors
                .iter()
                .flatten()
                .filter(|t| f.is_empty() || t.info.name.to_lowercase().contains(&f))
                .collect(),
            _ => Vec::new(),
        }
    }

    pub fn draw(&mut self, f: &mut Frame) {
        let [top, body, help] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .areas(f.area());
        let title = match &self.model {
            Some(p) => p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            None => "no model".into(),
        };
        let selected = TABS.iter().position(|(t, _)| *t == self.tab).unwrap_or(0);
        let [bar, rule] =
            Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(top);
        let [brand, tabs_area, name] = Layout::horizontal([
            Constraint::Length(14),
            Constraint::Min(40),
            Constraint::Percentage(40),
        ])
        .areas(bar);
        f.render_widget(Paragraph::new(" modelbuilder".bold()), brand);
        let tabs = Tabs::new(TABS.iter().map(|(_, l)| *l))
            .select(selected)
            .highlight_style(Style::new().bold().fg(SERIES_1));
        f.render_widget(tabs, tabs_area);
        f.render_widget(
            Paragraph::new(Line::styled(title, Style::new().fg(MUTED)).right_aligned()),
            name,
        );
        f.render_widget(
            Block::new()
                .borders(Borders::TOP)
                .border_style(Style::new().fg(MUTED)),
            rule,
        );
        match self.tab {
            Tab::Overview => self.draw_overview(f, body),
            Tab::Tensors => self.draw_tensors(f, body),
            Tab::Plan => self.draw_plan(f, body),
            Tab::Job => self.draw_job(f, body),
        }
        let keys = match self.tab {
            Tab::Tensors if self.filtering => {
                format!(" filter: {}▏  Enter keep · Esc clear", self.filter)
            }
            Tab::Tensors => " ↑↓ PgUp PgDn scroll · / filter · 1-4 tabs · q quit".into(),
            Tab::Plan => " ↑↓ PgUp PgDn scroll · 1-4 tabs · q quit".into(),
            Tab::Job => " c cancel job · 1-4 tabs · q quit".into(),
            Tab::Overview => " 1-4 or Tab to switch · q quit".into(),
        };
        f.render_widget(Paragraph::new(keys).style(Style::new().fg(MUTED)), help);
    }

    fn draw_overview(&self, f: &mut Frame, area: Rect) {
        let r = match &self.inspect {
            Err(e) => return message(f, area, &format!("Could not read the model: {e}"), CRITICAL),
            Ok(None) => {
                return message(
                    f,
                    area,
                    "No model. Run `modelbuilder tui <model.gguf | hf-dir>`.",
                    MUTED,
                )
            }
            Ok(Some(r)) => r,
        };
        let rep = &r.report;
        let a = &rep.architecture;
        let q = &rep.quantization;
        let kv = &rep.kv_cache;
        let [summary, rest] =
            Layout::vertical([Constraint::Length(10), Constraint::Min(0)]).areas(area);
        let [left, right] =
            Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .areas(summary);
        let kvline = |k: &str, v: String| {
            Line::from(vec![
                Span::styled(format!("{k:<14}"), Style::new().fg(MUTED)),
                Span::raw(v),
            ])
        };
        let arch = vec![
            kvline("family", a.family.clone().unwrap_or_else(|| "–".into())),
            kvline(
                "parameters",
                format!(
                    "{} ({} active/token)",
                    count(rep.params.total),
                    count(rep.params.active_per_token)
                ),
            ),
            kvline(
                "size",
                format!(
                    "{} · {:.2} bits/param",
                    bytes(q.total_bytes),
                    q.bits_per_param
                ),
            ),
            kvline(
                "layers",
                format!(
                    "{}{}",
                    a.num_layers,
                    if a.mtp_modules > 0 {
                        format!(" + {} MTP", a.mtp_modules)
                    } else {
                        String::new()
                    }
                ),
            ),
            kvline(
                "hidden/vocab",
                format!("{} / {}", count_opt(a.hidden_size), count_opt(a.vocab_size)),
            ),
            kvline("context", count_opt(a.max_positions)),
            kvline("pattern", a.layer_pattern.to_string()),
        ];
        f.render_widget(
            Paragraph::new(arch)
                .wrap(Wrap { trim: false })
                .block(section("Architecture")),
            left,
        );

        let mut quant: Vec<Line> = q
            .by_dtype
            .iter()
            .map(|d| {
                kvline(
                    &d.dtype,
                    format!(
                        "{:>5} tensors {:>9} {:>6.2} bits",
                        d.tensors,
                        bytes(d.bytes),
                        d.bits_per_param
                    ),
                )
            })
            .collect();
        if let Some(rot) = &q.rotation {
            quant.push(kvline(
                "rotation",
                format!("{} ({} tensors)", rot.scheme, rot.rotated_tensors),
            ));
        }
        quant.push(kvline(
            "KV cache",
            kv.precisions.first().map_or("–".into(), |p| {
                format!(
                    "{}/token {} · {} at {} tokens",
                    bytes(p.bytes_per_token as u64),
                    p.name,
                    bytes(p.bytes_at_context as u64),
                    count_opt(kv.context)
                )
            }),
        ));
        f.render_widget(
            Paragraph::new(quant).block(section("Quantization & KV")),
            right,
        );

        let [layers, params] =
            Layout::vertical([Constraint::Length(5), Constraint::Min(0)]).areas(rest);
        let strip: Vec<Span> = r
            .layers
            .iter()
            .map(|l| {
                let c = match l.layer.mixer {
                    Mixer::Attention(_) => SERIES_1,
                    Mixer::LinearAttention(_) => SERIES_2,
                    _ => MUTED,
                };
                Span::styled("█", Style::new().fg(c))
            })
            .collect();
        let legend = Line::from(vec![
            Span::styled("█ ", Style::new().fg(SERIES_1)),
            Span::raw("full attention  "),
            Span::styled("█ ", Style::new().fg(SERIES_2)),
            Span::raw("linear attention"),
        ]);
        f.render_widget(
            Paragraph::new(vec![Line::from(strip), legend])
                .wrap(Wrap { trim: false })
                .block(section("Layers")),
            layers,
        );

        let p = &rep.params;
        let parts = [
            ("embedding", p.embedding),
            ("LM head", p.lm_head),
            ("attention", p.attention),
            ("linear attention", p.linear_attention),
            ("dense FFN", p.ffn_dense),
            ("routed experts", p.moe_routed_experts),
            ("shared experts", p.moe_shared_experts),
            ("MTP", p.mtp),
            ("other", p.other + p.norms + p.router + p.multimodal),
        ];
        let max = parts.iter().map(|x| x.1).max().unwrap_or(1).max(1);
        let width = params.width.saturating_sub(34) as u64;
        let mut lines: Vec<Line> = parts
            .iter()
            .filter(|x| x.1 > 0)
            .map(|(k, v)| {
                let n = (v * width / max).max(1) as usize;
                Line::from(vec![
                    Span::styled(format!("{k:<17}"), Style::new().fg(MUTED)),
                    Span::styled("▇".repeat(n), Style::new().fg(SERIES_1)),
                    Span::raw(format!(" {}", count(*v))),
                ])
            })
            .collect();
        for w in &rep.warnings {
            lines.push(Line::from(vec![
                Span::styled("! ", Style::new().fg(WARNING).bold()),
                Span::raw(w.clone()),
            ]));
        }
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(section("Parameters")),
            params,
        );
    }

    fn draw_tensors(&mut self, f: &mut Frame, area: Rect) {
        let rows: Vec<Row> = self
            .tensor_rows()
            .into_iter()
            .map(|t| {
                Row::new(vec![
                    Cell::from(t.info.name.clone()),
                    Cell::from(fmt::dtype(&t.info.dtype)),
                    Cell::from(format!("{:?}", t.info.shape)),
                    Cell::from(Line::from(bytes(t.info.n_bytes)).right_aligned()),
                    Cell::from(
                        format!("{:?} · {:?}", t.role.component, t.role.kind).to_lowercase(),
                    ),
                ])
            })
            .collect();
        let n = rows.len();
        let table = Table::new(
            rows,
            [
                Constraint::Percentage(42),
                Constraint::Length(8),
                Constraint::Length(16),
                Constraint::Length(11),
                Constraint::Min(10),
            ],
        )
        .header(Row::new(["name", "dtype", "shape", "bytes", "role"]).style(Style::new().fg(MUTED)))
        .row_highlight_style(Style::new().add_modifier(Modifier::REVERSED))
        .block(section(&format!(
            "Tensors ({n}){}",
            if self.filter.is_empty() {
                String::new()
            } else {
                format!(" matching \"{}\"", self.filter)
            }
        )));
        f.render_stateful_widget(table, area, &mut self.tensors);
    }

    fn draw_plan(&self, f: &mut Frame, area: Rect) {
        let plan = match &self.plan {
            None if self.model.is_none() => {
                return message(f, area, "No model to plan for.", MUTED)
            }
            None => return message(f, area, "Planning the whole catalog…", MUTED),
            Some(Err(e)) => return message(f, area, e, CRITICAL),
            Some(Ok(p)) => p,
        };
        let mut lines: Vec<Line> = Vec::new();
        for fp in &plan.features {
            let (tone, label) = if !fp.compat.blockers.is_empty() {
                (CRITICAL, "✕ blocked")
            } else if !fp.compat.warnings.is_empty() {
                (WARNING, "! compatible, with warnings")
            } else {
                (GOOD, "✓ compatible")
            };
            let mut head = vec![
                Span::raw(fp.title).bold(),
                Span::styled(format!("  {}  ", fp.id), Style::new().fg(MUTED)),
                Span::styled(label, Style::new().fg(tone)),
            ];
            if let Some(e) = &fp.estimate {
                head.push(Span::styled(
                    format!(
                        "  · {:?} risk · confidence {:?}",
                        e.risk.level, e.confidence
                    )
                    .to_lowercase(),
                    Style::new().fg(MUTED),
                ));
            }
            lines.push(Line::from(head));
            lines.push(Line::styled(
                format!("  {}", fp.summary),
                Style::new().fg(MUTED),
            ));
            for b in fp.compat.blockers.iter().chain(&fp.compat.warnings) {
                lines.push(Line::from(format!("  • {b}")));
            }
            if let Some(e) = &fp.estimate {
                for x in &e.effects {
                    lines.push(Line::from(format!(
                        "  {}: {} → {}",
                        x.metric,
                        fmt::effect(x.before, &x.unit),
                        fmt::effect(x.after, &x.unit)
                    )));
                }
                for s in &e.stages {
                    lines.push(Line::from(format!(
                        "  stage {}: {} trainable, {}–{} tokens, {}",
                        s.name,
                        count(s.trainable_params),
                        count(s.tokens.low as u64),
                        count(s.tokens.high as u64),
                        s.loss
                    )));
                }
            }
            for c in &fp.compute {
                let (tone, fit) = match c.fits {
                    Fit::Yes => (GOOD, "✓ fits"),
                    Fit::WithPackedTrunk => (WARNING, "! packed trunk"),
                    Fit::No => (CRITICAL, "✕ doesn't fit"),
                    Fit::NotApplicable => (MUTED, "n/a"),
                };
                lines.push(Line::from(vec![
                    Span::raw(format!("    {:<14}", c.profile)),
                    Span::raw(format!(
                        "{:>18} GPU-h  {:>16} wall  {:>6.1} GiB  ",
                        range(c.gpu_hours.low, c.gpu_hours.high),
                        range(c.wall_hours.low, c.wall_hours.high),
                        c.peak_gib_per_gpu
                    )),
                    Span::styled(fit, Style::new().fg(tone)),
                ]));
            }
            lines.push(Line::raw(""));
        }
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .scroll((self.plan_scroll, 0))
                .block(section("Plan (default parameters)")),
            area,
        );
    }

    fn draw_job(&self, f: &mut Frame, area: Rect) {
        let Some(j) = &self.job else {
            let text = match &self.job_error {
                Some(e) => e.clone(),
                None => "No job. Run `modelbuilder tui --job runs/<name>/job.json` to train, or `--events events.jsonl` to follow a job running elsewhere.".into(),
            };
            let color = if self.job_error.is_some() {
                CRITICAL
            } else {
                MUTED
            };
            return message(f, area, &text, color);
        };
        let s = &j.summary;
        let (tone, status) = match s.status {
            JobStatus::Running => (MUTED, "• running"),
            JobStatus::Succeeded => (GOOD, "✓ succeeded"),
            JobStatus::Failed => (CRITICAL, "✕ failed"),
            JobStatus::Cancelled => (WARNING, "! cancelled"),
        };
        let progress = j.events.iter().rev().find_map(|e| match &e.event {
            Event::Progress {
                step,
                steps,
                loss,
                accuracy,
                tokens_per_s,
                ..
            } => Some((*step, *steps, *loss, *accuracy, *tokens_per_s)),
            _ => None,
        });
        let eval = j.events.iter().rev().find_map(|e| match &e.event {
            Event::Eval {
                step,
                loss,
                accuracy,
                ..
            } => Some((*step, *loss, *accuracy)),
            _ => None,
        });
        let [head, gauge, charts, bottom] = Layout::vertical([
            Constraint::Length(3),
            Constraint::Length(1),
            Constraint::Min(8),
            Constraint::Length(8),
        ])
        .areas(area);
        let mut info = vec![
            Span::raw(s.job_id.clone().unwrap_or_else(|| format!("job {}", s.id))).bold(),
            Span::raw("  "),
            Span::styled(status, Style::new().fg(tone)),
        ];
        if let Some((_, _, loss, acc, tps)) = progress {
            info.push(Span::raw(format!(
                "   train loss {loss:.4} · acc {}",
                pct(acc)
            )));
            if let Some(t) = tps {
                info.push(Span::styled(
                    format!(" · {} tok/s", count(t as u64)),
                    Style::new().fg(MUTED),
                ));
            }
        }
        if let Some((step, loss, acc)) = eval {
            info.push(Span::raw(format!(
                "   eval loss {loss:.4} · top-1 {} @ {step}",
                pct(Some(acc))
            )));
        }
        let mut head_lines = vec![Line::from(info)];
        if let Some(e) = &s.error {
            head_lines.push(Line::styled(e.clone(), Style::new().fg(CRITICAL)));
        }
        for (k, v) in &s.outputs {
            head_lines.push(Line::from(vec![
                Span::styled(format!("{k}: "), Style::new().fg(MUTED)),
                Span::raw(v.clone()),
            ]));
        }
        f.render_widget(Paragraph::new(head_lines), head);
        let ratio = progress.map_or(0.0, |(step, steps, ..)| step as f64 / steps.max(1) as f64);
        let label = progress.map_or("waiting for the first step".into(), |(step, steps, ..)| {
            format!("step {step}/{steps}")
        });
        f.render_widget(
            Gauge::default()
                .ratio(ratio.clamp(0.0, 1.0))
                .label(label)
                .gauge_style(Style::new().fg(SERIES_1)),
            gauge,
        );

        let series = |f: fn(&Event) -> Option<(f64, f64)>| -> Vec<(f64, f64)> {
            j.events.iter().filter_map(|e| f(&e.event)).collect()
        };
        let train_loss = series(|e| match e {
            Event::Progress { step, loss, .. } => Some((*step as f64, *loss)),
            _ => None,
        });
        let eval_loss = series(|e| match e {
            Event::Eval { step, loss, .. } => Some((*step as f64, *loss)),
            _ => None,
        });
        let train_acc = series(|e| match e {
            Event::Progress {
                step,
                accuracy: Some(a),
                ..
            } => Some((*step as f64, *a)),
            _ => None,
        });
        let eval_acc = series(|e| match e {
            Event::Eval { step, accuracy, .. } => Some((*step as f64, *accuracy)),
            _ => None,
        });
        let [lc, ac] = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
            .areas(charts);
        f.render_widget(chart("Loss", &train_loss, &eval_loss, None), lc);
        f.render_widget(
            chart("Top-1 accuracy", &train_acc, &eval_acc, Some((0.0, 1.0))),
            ac,
        );

        let recent: Vec<Line> = j
            .events
            .iter()
            .rev()
            .take(bottom.height.saturating_sub(2) as usize)
            .rev()
            .map(|e| Line::raw(event_line(e)))
            .chain(
                j.log
                    .iter()
                    .rev()
                    .take(2)
                    .rev()
                    .map(|l| Line::styled(l.clone(), Style::new().fg(MUTED))),
            )
            .collect();
        f.render_widget(
            Paragraph::new(recent).block(section(&format!("Events ({})", j.events.len()))),
            bottom,
        );
    }
}

fn next_tab(t: Tab, by: usize) -> Tab {
    let i = TABS.iter().position(|(x, _)| *x == t).unwrap_or(0);
    TABS[(i + by) % TABS.len()].0
}

fn count_opt(v: Option<u64>) -> String {
    v.map_or("–".into(), count)
}

fn section(title: &str) -> Block<'static> {
    Block::bordered()
        .title(Span::styled(format!(" {title} "), Style::new().bold()))
        .border_style(Style::new().fg(MUTED))
}

fn message(f: &mut Frame, area: Rect, text: &str, color: Color) {
    f.render_widget(
        Paragraph::new(Text::styled(text.to_string(), Style::new().fg(color)))
            .wrap(Wrap { trim: true })
            .block(section("")),
        area,
    );
}

fn chart<'a>(
    title: &'a str,
    train: &'a [(f64, f64)],
    eval: &'a [(f64, f64)],
    y: Option<(f64, f64)>,
) -> Chart<'a> {
    let all = train.iter().chain(eval);
    let (mut x0, mut x1, mut y0, mut y1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for (x, v) in all {
        x0 = x0.min(*x);
        x1 = x1.max(*x);
        y0 = y0.min(*v);
        y1 = y1.max(*v);
    }
    if x0 > x1 {
        (x0, x1, y0, y1) = (0.0, 1.0, 0.0, 1.0);
    }
    if x1 == x0 {
        x1 = x0 + 1.0;
    }
    let (y0, y1) = y.unwrap_or_else(|| {
        let pad = ((y1 - y0) * 0.05).max(1e-3);
        (y0 - pad, y1 + pad)
    });
    let label = |v: f64| {
        Span::styled(
            if y1 <= 1.0 && y0 >= 0.0 {
                format!("{:.0}%", v * 100.0)
            } else {
                format!("{v:.2}")
            },
            Style::new().fg(MUTED),
        )
    };
    let datasets = vec![
        Dataset::default()
            .name("train")
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::new().fg(SERIES_1))
            .data(train),
        Dataset::default()
            .name("eval")
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::new().fg(SERIES_2))
            .data(eval),
    ];
    Chart::new(datasets)
        .block(section(title))
        .x_axis(
            Axis::default()
                .bounds([x0, x1])
                .labels([
                    Span::styled(format!("{x0:.0}"), Style::new().fg(MUTED)),
                    Span::styled(format!("step {x1:.0}"), Style::new().fg(MUTED)),
                ])
                .style(Style::new().fg(MUTED)),
        )
        .y_axis(
            Axis::default()
                .bounds([y0, y1])
                .labels([label(y0), label(y1)])
                .style(Style::new().fg(MUTED)),
        )
}

fn event_line(e: &EventRecord) -> String {
    match &e.event {
        Event::Started {
            job_id,
            device,
            trainable_params,
            ..
        } => {
            format!(
                "started {job_id} on {device}, {} trainable",
                trainable_params.map_or("?".into(), count)
            )
        }
        Event::Progress {
            step,
            steps,
            loss,
            accuracy,
            ..
        } => format!(
            "step {step}/{steps}  loss {loss:.4}  acc {}",
            pct(*accuracy)
        ),
        Event::Eval {
            step,
            loss,
            accuracy,
            ..
        } => format!(
            "eval @ {step}  loss {loss:.4}  top-1 {}",
            pct(Some(*accuracy))
        ),
        Event::Checkpoint { step, path } => format!("checkpoint @ {step}: {path}"),
        Event::Finished { status, .. } => format!("finished: {status}"),
        Event::Error { message } => format!("error: {message}"),
    }
}

/// Runs the dashboard in the current terminal until the user quits.
pub fn run(opts: Options) -> std::io::Result<()> {
    let mut app = App::new(opts);
    let mut terminal = ratatui::init();
    let result = main_loop(&mut terminal, &mut app);
    ratatui::restore();
    result
}

fn main_loop(terminal: &mut DefaultTerminal, app: &mut App) -> std::io::Result<()> {
    let tick = Duration::from_millis(250);
    let mut last = Instant::now();
    while !app.quit {
        terminal.draw(|f| app.draw(f))?;
        let wait = tick.saturating_sub(last.elapsed());
        if event::poll(wait)? {
            if let TermEvent::Key(k) = event::read()? {
                app.on_key(k);
            }
        }
        if last.elapsed() >= tick {
            app.tick();
            last = Instant::now();
        }
    }
    Ok(())
}
