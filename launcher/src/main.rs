//! Mequa Launcher: a small native window that starts and stops the dashboard,
//! checks the Meta Model API key, shows live status and logs, and opens the dashboard in the browser.
//! Theme: navy base, blackish-blue depth, red highlights. Standard library plus eframe only.

#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

use eframe::egui::{self, Align2, Color32, FontId, Margin, RichText, Rounding, Sense, Stroke, Vec2};
use std::io::{BufRead, BufReader, Read};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const NAVY: Color32 = Color32::from_rgb(11, 26, 58);
const NAVY_2: Color32 = Color32::from_rgb(16, 35, 73);
const INK: Color32 = Color32::from_rgb(5, 9, 18);
const INK_2: Color32 = Color32::from_rgb(8, 14, 31);
const RED: Color32 = Color32::from_rgb(225, 29, 72);
const RED_HOT: Color32 = Color32::from_rgb(255, 59, 92);
const LINE: Color32 = Color32::from_rgb(29, 45, 87);
const TEXT: Color32 = Color32::from_rgb(232, 238, 255);
const MUTED: Color32 = Color32::from_rgb(143, 160, 200);
const GREEN: Color32 = Color32::from_rgb(34, 197, 94);

const DASH_PORT: u16 = 8501;
const MAX_LOG_LINES: usize = 3000;

fn meta_key_set() -> bool {
    std::env::var("META_API_KEY").map(|v| !v.trim().is_empty()).unwrap_or(false)
}

#[cfg(windows)]
fn hide(c: &mut Command) {
    use std::os::windows::process::CommandExt;
    c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
}
#[cfg(not(windows))]
fn hide(_c: &mut Command) {}

#[derive(Default)]
struct Status {
    dashboard: bool,
    meta_key: bool,
    python: Option<String>,
    python_checked: bool,
    streamlit_ok: bool,
}

type Logs = Arc<Mutex<Vec<String>>>;

fn port_open(port: u16) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    TcpStream::connect_timeout(&addr, Duration::from_millis(250)).is_ok()
}

fn resolve_python() -> Option<String> {
    if let Ok(p) = std::env::var("MEQUA_PYTHON") {
        if !p.trim().is_empty() {
            return Some(p);
        }
    }
    let cands: Vec<(&str, Vec<&str>)> = vec![("python", vec![]), ("python3", vec![]), ("py", vec!["-3"])];
    for (exe, pre) in cands {
        let mut c = Command::new(exe);
        c.args(&pre).args(["-c", "import sys;print(sys.executable)"]);
        hide(&mut c);
        if let Ok(o) = c.output() {
            if o.status.success() {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if !s.is_empty() {
                    return Some(s);
                }
            }
        }
    }
    None
}

fn find_root() -> Option<PathBuf> {
    let mut starts: Vec<PathBuf> = Vec::new();
    if let Ok(e) = std::env::current_exe() {
        if let Some(p) = e.parent() {
            starts.push(p.to_path_buf());
        }
    }
    if let Ok(c) = std::env::current_dir() {
        starts.push(c);
    }
    for s in starts {
        for a in s.ancestors() {
            if a.join("app.py").exists() && a.join("mequa").is_dir() {
                return Some(a.to_path_buf());
            }
        }
    }
    None
}

fn open_url(url: &str) {
    #[cfg(windows)]
    let mut c = {
        let mut c = Command::new("cmd");
        c.args(["/C", "start", "", url]);
        c
    };
    #[cfg(target_os = "macos")]
    let mut c = {
        let mut c = Command::new("open");
        c.arg(url);
        c
    };
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut c = {
        let mut c = Command::new("xdg-open");
        c.arg(url);
        c
    };
    hide(&mut c);
    let _ = c.spawn();
}

fn kill_tree(child: &mut Child) {
    #[cfg(windows)]
    {
        let mut c = Command::new("taskkill");
        c.args(["/PID", &child.id().to_string(), "/T", "/F"]);
        hide(&mut c);
        let _ = c.output();
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn push_log(logs: &Logs, line: String) {
    if let Ok(mut l) = logs.lock() {
        l.push(line);
        let n = l.len();
        if n > MAX_LOG_LINES {
            l.drain(0..n - MAX_LOG_LINES);
        }
    }
}

fn pipe_lines<R: Read + Send + 'static>(r: R, logs: Logs, ctx: egui::Context) {
    std::thread::spawn(move || {
        let mut rd = BufReader::new(r);
        let mut buf = Vec::new();
        loop {
            buf.clear();
            match rd.read_until(b'\n', &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let s = String::from_utf8_lossy(&buf).trim_end().to_string();
                    push_log(&logs, s);
                    ctx.request_repaint();
                }
            }
        }
    });
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Primary,
    Secondary,
    Danger,
}

/// Custom painted button: red gradient-look primary, navy secondary, outlined danger, with a hover glow.
fn action_button(ui: &mut egui::Ui, text: &str, kind: Kind, enabled: bool, width: f32) -> egui::Response {
    let size = Vec2::new(width, 42.0);
    let sense = if enabled { Sense::click() } else { Sense::hover() };
    let (rect, resp) = ui.allocate_exact_size(size, sense);
    let hovered = enabled && resp.hovered();
    let pressed = enabled && resp.is_pointer_button_down_on();
    let rounding = Rounding::same(12.0);
    let painter = ui.painter();

    let (fill, stroke, fg) = match kind {
        Kind::Primary => (if pressed { Color32::from_rgb(159, 18, 57) } else if hovered { RED_HOT } else { RED }, Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 40)), Color32::WHITE),
        Kind::Secondary => (if hovered { Color32::from_rgb(24, 48, 98) } else { NAVY_2 }, Stroke::new(1.0, if hovered { RED_HOT } else { Color32::from_rgb(42, 63, 120) }), Color32::from_rgb(219, 230, 255)),
        Kind::Danger => (if hovered { Color32::from_rgb(70, 10, 26) } else { INK_2 }, Stroke::new(1.5, RED), Color32::from_rgb(255, 180, 195)),
    };
    let (fill, stroke, fg) = if enabled {
        (fill, stroke, fg)
    } else {
        (Color32::from_rgb(14, 22, 44), Stroke::new(1.0, LINE), Color32::from_rgb(90, 104, 140))
    };
    if hovered && kind == Kind::Primary {
        painter.rect_filled(rect.expand(3.0), Rounding::same(15.0), Color32::from_rgba_unmultiplied(225, 29, 72, 60));
    }
    painter.rect_filled(rect, rounding, fill);
    painter.rect_stroke(rect, rounding, stroke);
    painter.text(rect.center(), Align2::CENTER_CENTER, text.to_uppercase(), FontId::proportional(13.5), fg);
    if enabled {
        resp.on_hover_cursor(egui::CursorIcon::PointingHand)
    } else {
        resp
    }
}

fn status_card(ui: &mut egui::Ui, title: &str, state: &str, ok: bool, detail: &str) {
    egui::Frame::none()
        .fill(NAVY)
        .stroke(Stroke::new(1.0, if ok { Color32::from_rgb(24, 90, 60) } else { LINE }))
        .rounding(Rounding::same(14.0))
        .inner_margin(Margin::same(14.0))
        .show(ui, |ui| {
            ui.set_min_height(74.0);
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                let (r, _) = ui.allocate_exact_size(Vec2::splat(12.0), Sense::hover());
                let c = if ok { GREEN } else { RED_HOT };
                if ok {
                    ui.painter().circle_filled(r.center(), 9.0, Color32::from_rgba_unmultiplied(34, 197, 94, 50));
                }
                ui.painter().circle_filled(r.center(), 5.0, c);
                ui.label(RichText::new(title.to_uppercase()).size(11.5).strong().color(MUTED));
            });
            ui.add_space(4.0);
            ui.label(RichText::new(state).size(19.0).strong().color(Color32::WHITE));
            ui.label(RichText::new(detail).size(11.5).color(MUTED));
        });
}

struct LauncherApp {
    status: Arc<Mutex<Status>>,
    logs: Logs,
    root: Option<PathBuf>,
    dash: Option<Child>,
    task: Option<(String, Child)>,
    open_when_ready: bool,
    ctx: egui::Context,
}

impl LauncherApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let ctx = cc.egui_ctx.clone();
        let mut v = egui::Visuals::dark();
        v.panel_fill = INK;
        v.window_fill = INK_2;
        v.extreme_bg_color = INK;
        v.override_text_color = Some(TEXT);
        v.selection.bg_fill = RED;
        v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, LINE);
        v.widgets.hovered.bg_stroke = Stroke::new(1.0, RED_HOT);
        ctx.set_visuals(v);
        let mut style = (*ctx.style()).clone();
        style.spacing.item_spacing = Vec2::new(10.0, 10.0);
        ctx.set_style(style);

        let status = Arc::new(Mutex::new(Status::default()));
        let logs: Logs = Arc::new(Mutex::new(Vec::new()));
        let root = find_root();
        match &root {
            Some(r) => push_log(&logs, format!("Project folder: {}", r.display())),
            None => push_log(&logs, "ERROR: could not find the Mequa folder (app.py and mequa/). Put the launcher inside it or start it from there.".into()),
        }

        // Python and Streamlit detection, once.
        {
            let st = status.clone();
            let lg = logs.clone();
            let c = ctx.clone();
            std::thread::spawn(move || {
                let py = resolve_python();
                let mut ok = false;
                if let Some(p) = &py {
                    let mut cmd = Command::new(p);
                    cmd.args(["-c", "import streamlit"]);
                    hide(&mut cmd);
                    ok = cmd.output().map(|o| o.status.success()).unwrap_or(false);
                    push_log(&lg, format!("Python: {}", p));
                    if !ok {
                        push_log(&lg, "ERROR: Streamlit is not installed for this Python. Run: python -m pip install -r requirements.txt".into());
                    }
                } else {
                    push_log(&lg, "ERROR: no Python found. Install Python 3 or set MEQUA_PYTHON to the full path of python.exe.".into());
                }
                if let Ok(mut s) = st.lock() {
                    s.python = py;
                    s.streamlit_ok = ok;
                    s.python_checked = true;
                }
                c.request_repaint();
            });
        }
        // Port poller (dashboard) plus Meta API key check.
        {
            let st = status.clone();
            let c = ctx.clone();
            std::thread::spawn(move || loop {
                let d = port_open(DASH_PORT);
                let k = meta_key_set();
                if let Ok(mut s) = st.lock() {
                    s.dashboard = d;
                    s.meta_key = k;
                }
                c.request_repaint();
                std::thread::sleep(Duration::from_millis(1500));
            });
        }
        if !meta_key_set() {
            push_log(&logs, "META_API_KEY is not set. Export it before launching the dashboard.".into());
        }

        Self { status, logs, root, dash: None, task: None, open_when_ready: false, ctx }
    }

    fn python(&self) -> Option<String> {
        self.status.lock().ok().and_then(|s| s.python.clone())
    }

    fn spawn_logged(&self, label: &str, mut cmd: Command) -> Option<Child> {
        if let Some(r) = &self.root {
            cmd.current_dir(r);
        }
        cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
        cmd.env("PYTHONUNBUFFERED", "1").env("PYTHONIOENCODING", "utf-8");
        hide(&mut cmd);
        match cmd.spawn() {
            Ok(mut child) => {
                push_log(&self.logs, format!("> {} started", label));
                if let Some(o) = child.stdout.take() {
                    pipe_lines(o, self.logs.clone(), self.ctx.clone());
                }
                if let Some(e) = child.stderr.take() {
                    pipe_lines(e, self.logs.clone(), self.ctx.clone());
                }
                Some(child)
            }
            Err(e) => {
                push_log(&self.logs, format!("ERROR: could not start {}: {}", label, e));
                None
            }
        }
    }

    fn start_dashboard(&mut self) {
        let Some(py) = self.python() else {
            push_log(&self.logs, "ERROR: Python was not found.".into());
            return;
        };
        if !meta_key_set() {
            push_log(&self.logs, "ERROR: META_API_KEY is not set. Export it, then launch the dashboard.".into());
            return;
        }
        let mut c = Command::new(py);
        c.args([
            "-m", "streamlit", "run", "app.py",
            "--server.port", &DASH_PORT.to_string(),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ]);
        self.dash = self.spawn_logged("dashboard", c);
        self.open_when_ready = self.dash.is_some();
    }

    fn stop_dashboard(&mut self) {
        if let Some(mut c) = self.dash.take() {
            kill_tree(&mut c);
            push_log(&self.logs, "> dashboard stopped".into());
        }
        self.open_when_ready = false;
    }

    fn run_task(&mut self, label: &str, args: &[&str]) {
        if self.task.is_some() {
            return;
        }
        let Some(py) = self.python() else {
            push_log(&self.logs, "ERROR: Python was not found.".into());
            return;
        };
        let mut c = Command::new(py);
        c.args(args);
        if let Some(ch) = self.spawn_logged(label, c) {
            self.task = Some((label.to_string(), ch));
        }
    }

    fn poll_children(&mut self) {
        if let Some(c) = self.dash.as_mut() {
            if let Ok(Some(code)) = c.try_wait() {
                push_log(&self.logs, format!("> dashboard exited ({})", code));
                self.dash = None;
                self.open_when_ready = false;
            }
        }
        let mut done: Option<(String, String)> = None;
        if let Some((label, c)) = self.task.as_mut() {
            if let Ok(Some(code)) = c.try_wait() {
                done = Some((label.clone(), code.to_string()));
            }
        }
        if let Some((label, code)) = done {
            push_log(&self.logs, format!("> {} finished ({})", label, code));
            self.task = None;
        }
    }
}

impl Drop for LauncherApp {
    fn drop(&mut self) {
        self.stop_dashboard();
        if let Some((_, mut c)) = self.task.take() {
            kill_tree(&mut c);
        }
    }
}

impl eframe::App for LauncherApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_children();
        ctx.request_repaint_after(Duration::from_millis(500));

        let (dash_up, key_set, py, py_checked, st_ok) = {
            let s = self.status.lock().unwrap();
            (s.dashboard, s.meta_key, s.python.clone(), s.python_checked, s.streamlit_ok)
        };

        if self.open_when_ready && dash_up {
            open_url(&format!("http://localhost:{}", DASH_PORT));
            self.open_when_ready = false;
        }

        let dash_url = format!("http://localhost:{}", DASH_PORT);
        let can_start = self.root.is_some() && py.is_some() && st_ok;

        egui::TopBottomPanel::bottom("footer")
            .frame(footer_frame())
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Research use only. No clinical guidance for any individual.").size(11.5).color(MUTED));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new(format!("Mequa Launcher {}", env!("CARGO_PKG_VERSION"))).size(11.5).color(MUTED));
                    });
                });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(INK).inner_margin(Margin::same(26.0)))
            .show(ctx, |ui| {
                // Header
                ui.horizontal(|ui| {
                    let (bar, _) = ui.allocate_exact_size(Vec2::new(5.0, 58.0), Sense::hover());
                    ui.painter().rect_filled(bar, Rounding::same(3.0), RED);
                    ui.vertical(|ui| {
                        ui.add_space(2.0);
                        ui.label(RichText::new("TISSUE MECHANICS WORKSPACE").size(11.5).strong().color(RED_HOT));
                        ui.label(RichText::new("MEQUA").size(34.0).strong().color(Color32::WHITE));
                    });
                });
                ui.add_space(8.0);

                // Status cards
                ui.columns(3, |cols| {
                    status_card(
                        &mut cols[0],
                        "Dashboard",
                        if dash_up { "Online" } else if self.dash.is_some() { "Starting" } else { "Offline" },
                        dash_up,
                        &dash_url,
                    );
                    status_card(
                        &mut cols[1],
                        "Meta API",
                        if key_set { "Key Set" } else { "Key Missing" },
                        key_set,
                        "META_API_KEY env",
                    );
                    let (state, detail) = if !py_checked {
                        ("Checking".to_string(), "looking for Python".to_string())
                    } else if py.is_none() {
                        ("Not Found".to_string(), "set MEQUA_PYTHON".to_string())
                    } else if !st_ok {
                        ("Streamlit Missing".to_string(), "pip install -r requirements.txt".to_string())
                    } else {
                        ("Ready".to_string(), py.clone().unwrap_or_default())
                    };
                    status_card(&mut cols[2], "Python", &state, py_checked && py.is_some() && st_ok, &detail);
                });
                ui.add_space(6.0);

                // Actions
                ui.horizontal_wrapped(|ui| {
                    if dash_up || self.dash.is_some() {
                        if action_button(ui, "Stop Dashboard", Kind::Danger, true, 190.0).clicked() {
                            self.stop_dashboard();
                        }
                    } else if action_button(ui, "Launch Dashboard", Kind::Primary, can_start, 190.0).clicked() {
                        self.start_dashboard();
                    }
                    if action_button(ui, "Open In Browser", Kind::Secondary, dash_up, 170.0).clicked() {
                        open_url(&dash_url);
                    }
                    if !key_set
                        && action_button(ui, "Why No Key?", Kind::Secondary, true, 170.0).clicked()
                    {
                        push_log(&self.logs, "Set META_API_KEY in your shell or .env, then relaunch the dashboard.".into());
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    let idle = self.task.is_none() && py.is_some();
                    if action_button(ui, "Environment Check", Kind::Secondary, idle, 190.0).clicked() {
                        self.run_task("environment check", &["-m", "mequa.doctor"]);
                    }
                    if action_button(ui, "Run Tests", Kind::Secondary, idle, 170.0).clicked() {
                        self.run_task("tests", &["-m", "pytest", "-q"]);
                    }
                    if action_button(ui, "Clear Log", Kind::Secondary, true, 150.0).clicked() {
                        if let Ok(mut l) = self.logs.lock() {
                            l.clear();
                        }
                    }
                    if let Some((label, _)) = &self.task {
                        ui.label(RichText::new(format!("running: {}", label)).color(RED_HOT).strong());
                    }
                });
                ui.add_space(6.0);

                // Log
                ui.label(RichText::new("LIVE LOG").size(11.5).strong().color(MUTED));
                egui::Frame::none()
                    .fill(INK_2)
                    .stroke(Stroke::new(1.0, LINE))
                    .rounding(Rounding::same(14.0))
                    .inner_margin(Margin::same(12.0))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
                            let lines: Vec<String> = self.logs.lock().map(|l| l.iter().rev().take(500).cloned().collect()).unwrap_or_default();
                            for line in lines.iter().rev() {
                                let low = line.to_lowercase();
                                let col = if low.contains("error") || low.contains("traceback") || low.contains("failed") {
                                    RED_HOT
                                } else if line.starts_with('>') {
                                    Color32::from_rgb(140, 200, 255)
                                } else {
                                    MUTED
                                };
                                ui.label(RichText::new(line).monospace().size(12.0).color(col));
                            }
                        });
                    });
            });
    }
}

fn footer_frame() -> egui::Frame {
    egui::Frame::none().fill(INK_2).inner_margin(Margin::symmetric(26.0, 8.0))
}

fn make_icon() -> egui::IconData {
    let n: usize = 64;
    let mut rgba = vec![0u8; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let (dx, dy) = (x as f32 - 31.5, y as f32 - 31.5);
            let r = (dx * dx + dy * dy).sqrt();
            let i = (y * n + x) * 4;
            let px: [u8; 4] = if r > 31.0 {
                [0, 0, 0, 0]
            } else if r > 20.0 {
                [11, 26, 58, 255]
            } else {
                [225, 29, 72, 255]
            };
            rgba[i..i + 4].copy_from_slice(&px);
        }
    }
    egui::IconData { rgba, width: n as u32, height: n as u32 }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Mequa Launcher")
            .with_icon(make_icon())
            .with_inner_size([860.0, 640.0])
            .with_min_inner_size([720.0, 520.0]),
        ..Default::default()
    };
    eframe::run_native("Mequa Launcher", options, Box::new(|cc| Ok(Box::new(LauncherApp::new(cc)))))
}
