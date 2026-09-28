use std::time::{Duration, Instant};

use mb_api::jobs::JobSource;
use mb_tui::{App, Options, Tab};
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Terminal;

fn screen(app: &mut App) -> String {
    let mut t = Terminal::new(TestBackend::new(140, 45)).unwrap();
    t.draw(|f| app.draw(f)).unwrap();
    let buf = t.backend().buffer().clone();
    let mut s = String::new();
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            s.push_str(buf[(x, y)].symbol());
        }
        s.push('\n');
    }
    s
}

fn key(app: &mut App, c: KeyCode) {
    app.on_key(KeyEvent::new(c, KeyModifiers::NONE));
}

fn settle(app: &mut App) {
    let t = Instant::now();
    while app.busy() && t.elapsed() < Duration::from_secs(20) {
        app.tick();
        std::thread::sleep(Duration::from_millis(20));
    }
    app.tick();
}

#[test]
fn shows_a_model_its_tensors_and_plan() {
    let dir = tempfile::tempdir().unwrap();
    let model = mb_fixtures::gguf_hybrid_ternary(dir.path());
    let mut app = App::new(Options {
        model: Some(model),
        hardware: vec!["1x24GB".into()],
        python: "python3".into(),
        ..Default::default()
    });
    let s = screen(&mut app);
    assert!(s.contains("qwen35"), "{s}");
    assert!(s.contains("Quantization"), "{s}");
    assert!(s.contains("linear attention"), "{s}");

    key(&mut app, KeyCode::Char('2'));
    assert_eq!(app.tab, Tab::Tensors);
    let s = screen(&mut app);
    assert!(s.contains("token_embd.weight"), "{s}");
    // Filter to one tensor.
    for c in "/output_norm".chars() {
        key(&mut app, KeyCode::Char(c));
    }
    key(&mut app, KeyCode::Enter);
    let s = screen(&mut app);
    assert!(s.contains("Tensors (1)"), "{s}");
    assert!(!s.contains("token_embd.weight"), "{s}");

    settle(&mut app);
    key(&mut app, KeyCode::Char('3'));
    let s = screen(&mut app);
    assert!(s.contains("kv-share") && s.contains("1x24GB"), "{s}");

    key(&mut app, KeyCode::Tab);
    assert_eq!(app.tab, Tab::Job);
    assert!(screen(&mut app).contains("No job"));
    key(&mut app, KeyCode::Char('q'));
    assert!(app.quit);
}

#[test]
fn follows_a_job_from_its_events_file() {
    let dir = tempfile::tempdir().unwrap();
    let events = dir.path().join("events.jsonl");
    std::fs::write(
        &events,
        include_str!("../../../schema/examples/events.jsonl"),
    )
    .unwrap();
    let mut app = App::new(Options {
        job: Some(JobSource::Events {
            events_path: events.display().to_string(),
        }),
        python: "python3".into(),
        ..Default::default()
    });
    assert_eq!(app.tab, Tab::Job, "with only a job, it opens on the job");
    settle(&mut app);
    let s = screen(&mut app);
    assert!(s.contains("succeeded"), "{s}");
    assert!(s.contains("mtp-align-example"), "{s}");
    assert!(s.contains("eval loss 1.1000"), "{s}");
    assert!(s.contains("Loss") && s.contains("Top-1 accuracy"), "{s}");

    // Without a model, the other tabs say so instead of failing.
    key(&mut app, KeyCode::Char('1'));
    assert!(screen(&mut app).contains("No model"));
}

#[test]
fn reports_an_unreadable_model() {
    let mut app = App::new(Options {
        model: Some("/no/such.gguf".into()),
        python: "python3".into(),
        ..Default::default()
    });
    assert!(screen(&mut app).contains("Could not read the model"));
}
