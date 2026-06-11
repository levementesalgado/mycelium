use eframe::egui;

pub struct PipelineState {
    pub active_stage: usize,
}

impl Default for PipelineState {
    fn default() -> Self {
        Self { active_stage: 0 }
    }
}

const STAGES: &[(&str, &str, &str)] = &[
    ("STFT", "📊", "Wavelet → STFT → 8 features espectrais"),
    ("ESN", "🌀", "Reservoir (ESN) → estados de alta dimensão"),
    ("CNN", "🧠", "Estados ESN reshape 2D → Conv2d → classificação"),
];

impl PipelineState {
    pub fn render(&mut self, ui: &mut egui::Ui, _params: &super::AllParams, mode: &str) {
        egui::Frame::NONE.fill(egui::Color32::from_gray(15)).show(ui, |ui| {
            let w = ui.available_width();
            let h = 90.0;
            let rect = ui.allocate_space(egui::vec2(w, h)).1;
            let n = STAGES.len();
            let box_w = (rect.width() / n as f32).min(180.0);
            let box_h = 60.0;
            let start_x = rect.min.x + (rect.width() - box_w * n as f32) / 2.0;

            let active_max = match mode {
                "reservoir" => 1,
                "cnn" => 2,
                _ => 0,
            };

            for (i, (name, icon, desc)) in STAGES.iter().enumerate() {
                let x = start_x + i as f32 * box_w;
                let y = rect.min.y + (rect.height() - box_h) / 2.0;
                let stage_rect = egui::Rect::from_min_size(
                    egui::pos2(x, y), egui::vec2(box_w - 8.0, box_h),
                );
                let is_active = i <= active_max;
                let is_current = i == active_max;
                let bg = if is_current {
                    egui::Color32::from_rgb(20, 50, 30)
                } else if is_active {
                    egui::Color32::from_gray(30)
                } else {
                    egui::Color32::from_gray(18)
                };

                ui.painter().rect_filled(stage_rect, 4.0, bg);
                let stroke_c = if is_current { egui::Color32::from_rgb(0, 200, 100) } else { egui::Color32::from_gray(50) };
                ui.painter().rect_stroke(stage_rect, 4.0, egui::Stroke::new(1.0, stroke_c), egui::StrokeKind::Outside);
                ui.painter().text(
                    egui::pos2(stage_rect.min.x + 8.0, stage_rect.min.y + 8.0),
                    egui::Align2::LEFT_TOP,
                    format!("{} {}", icon, name),
                    egui::TextStyle::Button.resolve(ui.style()),
                    egui::Color32::from_rgb(200, 200, 200),
                );
                ui.painter().text(
                    egui::pos2(stage_rect.min.x + 8.0, stage_rect.min.y + 30.0),
                    egui::Align2::LEFT_TOP,
                    *desc,
                    egui::TextStyle::Monospace.resolve(ui.style()),
                    egui::Color32::from_gray(120),
                );
                if i < n - 1 {
                    let arrow_x = x + box_w - 4.0;
                    let arrow_y = rect.center().y;
                    let arrow_end = x + box_w + 4.0;
                    ui.painter().line_segment(
                        [egui::pos2(arrow_x, arrow_y), egui::pos2(arrow_end, arrow_y)],
                        egui::Stroke::new(2.0, egui::Color32::from_gray(60)),
                    );
                    ui.painter().text(
                        egui::pos2(arrow_end, arrow_y - 6.0),
                        egui::Align2::LEFT_CENTER,
                        "▶",
                        egui::TextStyle::Monospace.resolve(ui.style()),
                        egui::Color32::from_gray(60),
                    );
                }
            }
        });
    }
}
