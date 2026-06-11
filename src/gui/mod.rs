use eframe::egui::{self, Color32, Frame, ScrollArea, CentralPanel};
use std::path::Path;
use crate::classify::{self, RawSegment, ClassifiedSegment};
use crate::interpret::Report;
use crate::pipeline::{Pipeline, PipelineMode};

pub mod params;
pub mod pipeline;
pub use params::*;
pub use pipeline::*;

pub struct MyceliumGui {
    pub params: AllParams,
    pub pipe_vis: PipelineState,
    pub pipe: Pipeline,
    pub image_list: Vec<String>,
    pub selected_image: Option<String>,
    pub image_texture: Option<egui::TextureHandle>,
    pub raw_segments: Vec<RawSegment>,
    pub classified: Vec<ClassifiedSegment>,
    pub selected_segment: usize,
    pub user_labels: Vec<(usize, String)>,
    pub label_input: String,
    pub current_report: Option<Report>,
    pub status_message: String,
    pub loaded_csv: String,
    pub show_params: bool,
    pub mode: String,
}

impl Default for MyceliumGui {
    fn default() -> Self {
        Self {
            params: AllParams::default(),
            pipe_vis: PipelineState::default(),
            pipe: Pipeline::classical(1.0),
            image_list: Vec::new(),
            selected_image: None,
            image_texture: None,
            raw_segments: Vec::new(),
            classified: Vec::new(),
            selected_segment: 0,
            user_labels: Vec::new(),
            label_input: String::new(),
            current_report: None,
            status_message: String::new(),
            loaded_csv: String::new(),
            show_params: true,
            mode: "classical".into(),
        }
    }
}

impl MyceliumGui {
    pub fn new() -> Self { Self::default() }

    fn scan_images(&mut self) {
        let mut images = Vec::new();
        for dir in ["dados_test", "dados_real", "dados"] {
            let d = Path::new(dir);
            if !d.is_dir() { continue; }
            if let Ok(entries) = std::fs::read_dir(d) {
                for e in entries.flatten() {
                    let p = e.path();
                    if p.extension().is_some_and(|x| x == "png") {
                        if let Some(name) = p.file_name() {
                            images.push(format!("{}/{}", dir, name.to_string_lossy()));
                        }
                    }
                }
            }
        }
        images.sort();
        self.image_list = images;
    }

    fn load_image(&mut self, ctx: &egui::Context, filename: &str) {
        match image::open(Path::new(filename)) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let size = [rgba.width() as usize, rgba.height() as usize];
                let pixels = rgba.into_raw();
                let ci = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);
                let tex = ctx.load_texture(filename, ci, egui::TextureOptions::LINEAR);
                self.image_texture = Some(tex);
                self.selected_image = Some(filename.to_string());
            }
            Err(e) => self.status_message = format!("Erro imagem: {}", e),
        }
    }

    fn load_csv_data(&mut self, path: &str) {
        match classify::load_csv(Path::new(path)) {
            Ok(segs) => {
                self.raw_segments = segs;
                self.selected_segment = 0;
                self.user_labels.clear();
                self.classified.clear();
                self.pipe = self.make_pipeline();
                self.loaded_csv = path.to_string();
                self.status_message = format!("Carregado: {}", path);
            }
            Err(e) => self.status_message = format!("Erro CSV: {}", e),
        }
    }

    fn make_pipeline(&self) -> Pipeline {
        let mode = match self.mode.as_str() {
            "reservoir" => PipelineMode::Reservoir,
            "cnn" => PipelineMode::Cnn,
            "rf" | "random_forest" | "randomforest" => PipelineMode::RandomForest,
            _ => PipelineMode::Classical,
        };
        let n_res = self.params.rc.n_reservoir as usize;
        let sr = self.params.rc.spectral_radius;
        match mode {
            PipelineMode::Classical => Pipeline::classical(self.params.rc.ridge_lambda),
            PipelineMode::Reservoir => Pipeline::reservoir(n_res, sr, self.params.rc.ridge_lambda),
            PipelineMode::Cnn => Pipeline::cnn(n_res, sr, self.params.cnn.learning_rate),
            PipelineMode::RandomForest => Pipeline::random_forest(),
        }
    }

    fn load_test_csv(&mut self) {
        let p = "dados_test/sinais_timecourse.csv";
        if Path::new(p).exists() {
            self.load_csv_data(p);
        } else {
            self.status_message = "dados_test/sinais_timecourse.csv não encontrado".into();
        }
    }

    fn load_real_csv(&mut self) {
        let d = Path::new("dados_real");
        if !d.is_dir() {
            self.status_message = "dados_real/ não existe".into();
            return;
        }
        let mut found: Option<String> = None;
        if let Ok(entries) = std::fs::read_dir(d) {
            for e in entries.flatten() {
                let p = e.path();
                if p.extension().is_some_and(|x| x == "csv") {
                    found = p.to_str().map(|s| s.to_string());
                    break;
                }
            }
        }
        match found {
            Some(p) => self.load_csv_data(&p),
            None => self.status_message = "Nenhum CSV em dados_real/".into(),
        }
    }

    fn train_from_labels(&mut self) {
        if self.user_labels.is_empty() {
            self.status_message = "Nenhum rótulo para treinar".into();
            return;
        }
        // Rebuild pipeline with current params
        self.pipe = self.make_pipeline();

        let feats: Vec<Vec<f64>> = self.user_labels.iter()
            .map(|(idx, _)| {
                let seg = &self.raw_segments[*idx];
                self.pipe.extract_features(&seg.signal)
            }).collect();
        let labels: Vec<String> = self.user_labels.iter().map(|(_, l)| l.clone()).collect();
        self.pipe.classifier.fit(&feats, &labels);
        self.pipe.classes = self.pipe.classifier.classes.clone();
        self.pipe.trained = true;

        self.classified = self.pipe.classify(&self.raw_segments);
        let acc = self.pipe.accuracy(&self.classified);
        let msg = format!("Treinado: modo={} acc={:.1}% classes={}",
            self.mode, acc * 100.0, self.pipe.classes.len());
        self.status_message = msg;
    }

    fn reset_model(&mut self) {
        self.pipe = self.make_pipeline();
        self.classified.clear();
        self.user_labels.clear();
        self.current_report = None;
        self.status_message = "Modelo resetado".into();
    }
}

impl eframe::App for MyceliumGui {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        CentralPanel::default().show(&ctx, |ui| {
            // ─── Top bar ────────────────────────────────
            ui.horizontal(|ui| {
                ui.heading("🍄 mycelium-net");
                ui.separator();
                egui::widgets::global_theme_preference_buttons(ui);
                ui.separator();
                ui.label(format!("fs={:.0}Hz | seg={}",
                    self.params.signal.sample_rate, self.raw_segments.len()));
                if !self.loaded_csv.is_empty() {
                    ui.separator();
                    ui.monospace(&self.loaded_csv);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.checkbox(&mut self.show_params, "⚙ Params");
                });
            });

            // ─── Action bar ─────────────────────────────
            ui.horizontal(|ui| {
                if ui.button("📂 Teste").clicked() { self.load_test_csv(); }
                if ui.button("📂 Real").clicked() { self.load_real_csv(); }
                if ui.button("🖼 Imagens").clicked() { self.scan_images(); }
                ui.separator();
                // Mode selector
                ui.label("Modo:");
                egui::ComboBox::from_id_salt("mode_sel")
                    .selected_text(&self.mode)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.mode, "classical".into(), "Clássico (STFT)");
                        ui.selectable_value(&mut self.mode, "reservoir".into(), "Reservoir (ESN)");
                        ui.selectable_value(&mut self.mode, "cnn".into(), "CNN (ESN+CNN)");
                    });
                if ui.button("🏋 Treinar").clicked() { self.train_from_labels(); }
                if ui.button("🔄 Reset").clicked() { self.reset_model(); }
                if !self.status_message.is_empty() {
                    ui.separator();
                    ui.label(&self.status_message);
                }
            });

            // ─── Pipeline nav (horizontal) ──────────────
            let modes = ["classical", "reservoir", "cnn"];
            let labels = ["⚡ Clássico", "🌀 Reservoir", "🧠 CNN"];
            ui.horizontal(|ui| {
                for (i, (label, m)) in labels.iter().zip(modes.iter()).enumerate() {
                    let sel = self.mode == *m;
                    let mut txt = egui::RichText::new(*label);
                    if sel { txt = txt.color(Color32::from_rgb(0, 220, 100)).strong(); }
                    if ui.selectable_label(sel, txt).clicked() {
                        self.mode = m.to_string();
                        self.pipe_vis.active_stage = i;
                        self.pipe = self.make_pipeline();
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.pipe.trained {
                        ui.label(format!("📈 {}", self.pipe.summary()));
                    }
                });
            });
            ui.separator();

            // ─── Main content area (2 columns) ──────────
            ui.columns(2, |cols| {
                cols[0].vertical(|ui| {
                    ScrollArea::vertical().id_source("col_left").show(ui, |ui| {
                        self.signal_plot(ui);
                        ui.separator();
                        self.images_panel(ui);
                        ui.separator();
                        self.labeling_panel(ui);
                    });
                });

                cols[1].vertical(|ui| {
                    ScrollArea::vertical().id_source("col_right").show(ui, |ui| {
                        self.pipe_vis.render(ui, &self.params, &self.mode);
                        ui.separator();
                        self.results_panel(ui);
                        if self.show_params {
                            ui.separator();
                            self.params_panel(ui);
                        }
                    });
                });
            });
        });

        ctx.request_repaint();
    }
}

// ─── Painéis ──────────────────────────────────────────────

impl MyceliumGui {
    fn params_panel(&mut self, ui: &mut egui::Ui) {
        Frame::group(ui.style()).show(ui, |ui| {
            ui.label("Reservoir");
            ui.add(egui::Slider::new(&mut self.params.rc.n_reservoir, 16.0..=4096.0).text("N").logarithmic(true));
            ui.add(egui::Slider::new(&mut self.params.rc.leak_rate, 0.01..=1.0).text("leak"));
            ui.add(egui::Slider::new(&mut self.params.rc.spectral_radius, 0.1..=2.0).text("ρ"));
            ui.add(egui::Slider::new(&mut self.params.rc.input_scale, 0.01..=10.0).text("in_scale").logarithmic(true));
            ui.add(egui::Slider::new(&mut self.params.rc.ridge_lambda, 1e-6..=1.0).text("λ").logarithmic(true));
        });
        Frame::group(ui.style()).show(ui, |ui| {
            ui.label("STFT / Wavelet");
            ui.add(egui::Slider::new(&mut self.params.signal.window_size, 16.0..=2048.0).text("window").logarithmic(true));
            ui.add(egui::Slider::new(&mut self.params.signal.hop_length, 4.0..=1024.0).text("hop").logarithmic(true));
            ui.add(egui::Slider::new(&mut self.params.signal.n_fft, 16.0..=4096.0).text("n_fft").logarithmic(true));
            ui.add(egui::Slider::new(&mut self.params.signal.sample_rate, 10.0..=100000.0).text("fs(Hz)").logarithmic(true));
        });
        Frame::group(ui.style()).show(ui, |ui| {
            ui.label("CNN");
            ui.add(egui::Slider::new(&mut self.params.cnn.conv1_channels, 4.0..=256.0).text("C1").logarithmic(true));
            ui.add(egui::Slider::new(&mut self.params.cnn.conv2_channels, 4.0..=256.0).text("C2").logarithmic(true));
            ui.add(egui::Slider::new(&mut self.params.cnn.kernel_size, 1.0..=7.0).text("K"));
            ui.add(egui::Slider::new(&mut self.params.cnn.learning_rate, 1e-6..=1.0).text("lr").logarithmic(true));
            ui.add(egui::Slider::new(&mut self.params.cnn.dropout, 0.0..=0.9).text("dropout"));
        });
        Frame::group(ui.style()).show(ui, |ui| {
            ui.label("ML");
            ui.add(egui::Slider::new(&mut self.params.ml.n_estimators, 1.0..=500.0).text("trees").logarithmic(true));
            ui.add(egui::Slider::new(&mut self.params.ml.max_depth, 1.0..=50.0).text("depth"));
            ui.add(egui::Slider::new(&mut self.params.ml.svm_cost, 0.01..=100.0).text("C").logarithmic(true));
            ui.add(egui::Slider::new(&mut self.params.ml.svm_gamma, 0.001..=10.0).text("γ").logarithmic(true));
            ui.add(egui::Slider::new(&mut self.params.ml.wavelet_threshold, 0.0..=2.0).text("w_thresh"));
        });
        Frame::group(ui.style()).show(ui, |ui| {
            ui.label("Training");
            ui.add(egui::Slider::new(&mut self.params.training.epochs, 1.0..=10000.0).text("epochs").logarithmic(true));
            ui.add(egui::Slider::new(&mut self.params.training.batch_size, 1.0..=512.0).text("batch").logarithmic(true));
            ui.add(egui::Slider::new(&mut self.params.training.val_split, 0.05..=0.5).text("val%"));
            ui.add(egui::Slider::new(&mut self.params.training.early_stop_patience, 1.0..=100.0).text("patience"));
            ui.add(egui::Slider::new(&mut self.params.training.l2_reg, 1e-8..=1.0).text("L2").logarithmic(true));
        });
    }

    fn signal_plot(&mut self, ui: &mut egui::Ui) {
        if self.raw_segments.is_empty() {
            ui.label("Nenhum dado. Clique 📂 Teste ou 📂 Real.");
            return;
        }
        let idx = self.selected_segment.min(self.raw_segments.len() - 1);
        let seg = &self.raw_segments[idx];
        let f = &seg.features;

        ui.horizontal(|ui| {
            if ui.button("◀").clicked() && self.selected_segment > 0 { self.selected_segment -= 1; }
            ui.label(format!("Seg {}/{} | {:.1}s–{:.1}s", idx + 1, self.raw_segments.len(), seg.time_start / 1000.0, seg.time_end / 1000.0));
            if ui.button("▶").clicked() && self.selected_segment < self.raw_segments.len() - 1 { self.selected_segment += 1; }
        });
        ui.label(format!("dom={:.0}Hz amp={:.1}µV spikes={:.1}/s ent={:.3} burst={:.2}",
            f.dominant_freq, f.mean_amplitude, f.spike_rate, f.spectral_entropy, f.burst_index));

        let w = ui.available_width().max(100.0);
        let (_, rect) = ui.allocate_space(egui::vec2(w, 80.0));
        let signal = &seg.signal;
        let n = signal.len().min(600);
        let step = signal.len().max(1) / n.max(1);
        let sampled: Vec<f64> = signal.iter().step_by(step).copied().collect();
        let m = sampled.len();
        if m > 1 {
            let mn = sampled.iter().cloned().fold(f64::MAX, f64::min);
            let mx = sampled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = (mx - mn).max(1.0);
            let pts: Vec<egui::Pos2> = sampled.iter().enumerate().map(|(i, &v)| {
                egui::pos2(
                    rect.min.x + (i as f32 / m as f32) * rect.width(),
                    rect.center().y - ((v - mn) / range * 0.8 - 0.4) as f32 * rect.height(),
                )
            }).collect();
            ui.painter().add(egui::Shape::line(pts, egui::Stroke::new(1.5, Color32::from_rgb(0, 200, 80))));
        }
    }

    fn images_panel(&mut self, ui: &mut egui::Ui) {
        if self.image_list.is_empty() { self.scan_images(); }
        egui::CollapsingHeader::new("🖼 Imagens").default_open(true).show(ui, |ui| {
            if self.image_list.is_empty() {
                ui.label("Nenhuma. Execute: Rscript R/generate_signals.R");
                return;
            }
            let list = self.image_list.clone();
            for row in list.chunks(3) {
                ui.horizontal(|ui| {
                    for name in row {
                        let sel = self.selected_image.as_deref() == Some(name);
                        let mut btn = egui::Button::new(egui::RichText::new(name).size(8.0));
                        if sel { btn = btn.fill(Color32::from_rgb(30, 60, 40)); }
                        if ui.add(btn).clicked() {
                            self.load_image(ui.ctx(), name);
                        }
                    }
                });
            }
            if let Some(tex) = &self.image_texture {
                ui.separator();
                let max_w = ui.available_width().min(500.0);
                let a = tex.size()[0] as f32 / tex.size()[1] as f32;
                let h = (max_w / a).min(300.0);
                ui.add(egui::Image::from_texture(egui::load::SizedTexture::new(tex.id(), egui::vec2(max_w, h))));
                if let Some(n) = &self.selected_image {
                    ui.monospace(n);
                }
            }
        });
    }

    fn labeling_panel(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("🏷 Rotulagem").default_open(true).show(ui, |ui| {
            if self.raw_segments.is_empty() { ui.label("Carregue CSV primeiro."); return; }
            let n = self.raw_segments.len();
            let idx = self.selected_segment.min(n - 1);
            let seg = &self.raw_segments[idx];
            let f = &seg.features;
            ui.label(format!("Seg {} | {:.1}s–{:.1}s | dom={:.0}Hz amp={:.1}µV",
                idx, seg.time_start / 1000.0, seg.time_end / 1000.0, f.dominant_freq, f.mean_amplitude));
            if !seg.ground_truth.is_empty() {
                ui.label(format!("GT: {}", seg.ground_truth));
            }
            let existing = self.user_labels.iter().find(|(i, _)| *i == idx).map(|(_, l)| l.clone());
            let hint = existing.as_deref().unwrap_or("");
            ui.horizontal(|ui| {
                ui.label("Rótulo:");
                let resp = ui.add(egui::TextEdit::singleline(&mut self.label_input).desired_width(120.0).hint_text(hint));
                if (resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) || ui.button("💾").clicked() {
                    if !self.label_input.is_empty() {
                        self.user_labels.push((idx, self.label_input.clone()));
                        self.label_input.clear();
                    }
                }
                if !hint.is_empty() { ui.colored_label(Color32::from_rgb(100, 200, 120), hint); }
            });
            ui.horizontal(|ui| {
                if ui.button("◀ Anterior").clicked() && self.selected_segment > 0 { self.selected_segment -= 1; }
                if ui.button("Próximo ▶").clicked() && self.selected_segment < n - 1 { self.selected_segment += 1; }
            });
            if !self.user_labels.is_empty() {
                ui.separator();
                let snap = self.user_labels.clone();
                let mut remove: Option<usize> = None;
                for (i, l) in &snap {
                    ui.horizontal(|ui| {
                        ui.monospace(format!("Seg.{}", i));
                        ui.label(format!("→ {}", l));
                        if ui.button("✕").clicked() { remove = Some(*i); }
                    });
                }
                if let Some(r) = remove { self.user_labels.retain(|(i, _)| i != &r); }
            }
        });
    }

    fn results_panel(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("🎯 Resultados").default_open(true).show(ui, |ui| {
            if !self.pipe.trained {
                ui.label("Rotule e clique 🏋 Treinar.");
                return;
            }
            if self.classified.is_empty() { return; }
            let acc = self.pipe.accuracy(&self.classified);
            ui.label(format!("Modo: {} | Classes: {} | Acc: {:.1}%",
                self.pipe.summary(), self.pipe.classes.len(), acc * 100.0));
            let list = self.classified.clone();
            for (i, seg) in list.iter().enumerate() {
                let bg = if i == self.selected_segment { Color32::from_rgb(20, 40, 30) } else { Color32::from_gray(18) };
                let icon = if !seg.ground_truth.is_empty() && seg.correct { "✓" } else { "→" };
                Frame::none().fill(bg).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.monospace(format!("{} [{:.1}s–{:.1}s] {} ({:.0}%)",
                            icon, seg.time_start / 1000.0, seg.time_end / 1000.0, seg.predicted_class, seg.confidence * 100.0));
                        if ui.button("📋").clicked() {
                            self.selected_segment = i;
                            self.current_report = Some(seg.report.clone());
                        }
                    });
                });
            }
            if let Some(r) = &self.current_report {
                ui.separator();
                Frame::none().fill(Color32::from_gray(20)).show(ui, |ui| {
                    ui.label("Observações:");
                    for o in &r.observations { ui.label(format!("  • {}", o)); }
                    ui.separator();
                    ui.label("Sumário:");
                    ui.label(&r.summary);
                });
            }
        });
    }
}
