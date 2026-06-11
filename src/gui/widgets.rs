// ─── Widgets Customizados ───────────────────────────────────
//
// Componentes reutilizáveis pra interface. Nada de biblioteca
// mágica — só egui puro e parámetros que você controla.
//
// Referência: Death March kara Hajimaru Isekai Kyousoukyoku
// — cada widget é uma skill. Sozinho é fraco, combinado vira
// build overpower.
//
// pitch~ pitch~ (som do ponteiro deslizando no slider)

// (por enquanto os widgets são inline no gui.rs)
// Este módulo é reservado para expansão futura:
// - Knob circular (estilo VST)
// - Waterfall plot (STFT em tempo real)
// - Gauge de performance

use eframe::egui;

/// Knob estilo VST (analógico). Input visual com
/// sensação tátil simulada.
pub struct Knob {
    value: f64,
    min: f64,
    max: f64,
    label: String,
}

impl Knob {
    pub fn new(value: f64, min: f64, max: f64, label: &str) -> Self {
        Self { value, min, max, label: label.to_string() }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) -> f64 {
        let size = 64.0;
        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(size, size + 20.0),
            egui::Sense::click_and_drag(),
        );

        let pct = ((self.value - self.min) / (self.max - self.min)) as f32;
        let angle = -0.75 * std::f32::consts::PI + pct * 1.5 * std::f32::consts::PI;
        let cx = rect.min.x + size / 2.0;
        let cy = rect.min.y + size / 2.0;
        let r = size / 2.0 - 6.0;

        // fundo
        ui.painter().circle_filled(
            egui::pos2(cx, cy),
            r + 2.0,
            egui::Color32::from_gray(30),
        );

        // arco
        let n_arc = 20;
        for i in 0..n_arc {
            let t = i as f32 / n_arc as f32;
            let a = -0.75 * std::f32::consts::PI + t * 1.5 * std::f32::consts::PI;
            let inner_r = r * 0.7;
            let is_on = t <= pct;
            let color = if is_on { egui::Color32::from_rgb(0, 200, 100) } else { egui::Color32::from_gray(50) };
            ui.painter().line_segment(
                [
                    egui::pos2(cx + a.cos() * inner_r, cy + a.sin() * inner_r),
                    egui::pos2(cx + a.cos() * r, cy + a.sin() * r),
                ],
                egui::Stroke::new(3.0, color),
            );
        }

        // ponteiro
        ui.painter().line_segment(
            [egui::pos2(cx, cy), egui::pos2(
                cx + angle.cos() * r * 0.8,
                cy + angle.sin() * r * 0.8,
            )],
            egui::Stroke::new(2.0, egui::Color32::WHITE),
        );

        // label
        ui.painter().text(
            egui::pos2(cx, rect.max.y),
            egui::Align2::CENTER_TOP,
            &self.label,
            egui::TextStyle::Small.resolve(ui.style()),
            egui::Color32::from_gray(120),
        );

        // drag
        if response.dragged_by(egui::PointerButton::Primary) {
            let delta = response.drag_delta().x - response.drag_delta().y;
            let step = (self.max - self.min) / 500.0;
            self.value = (self.value + delta as f64 * step).clamp(self.min, self.max);
        }

        self.value
    }
}
