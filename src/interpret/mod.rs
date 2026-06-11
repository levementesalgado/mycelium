// ─── Interpretação: gerador de relatório descritivo ────
//
// Gera um relatório a partir das features extraídas do sinal
// e da classificação fornecida pelo modelo treinado pelo
// usuário. NENHUMA descrição é hardcoded — o modelo aprende
// o que o usuário ensina.
//
// "O micélio não sabe o que é 'estresse hídrico'. Quem sabe
//  é você, depois de fazer o experimento."
//
// kyun~ (som de feature sendo medida)

#[derive(Clone, Debug)]
pub struct SignalFeatures {
    pub dominant_freq: f64,
    pub mean_amplitude: f64,
    pub peak_amplitude: f64,
    pub spike_rate: f64,
    pub spectral_centroid: f64,
    pub spectral_entropy: f64,
    pub burst_index: f64,
    pub coherence: f64,
}

impl Default for SignalFeatures {
    fn default() -> Self {
        Self {
            dominant_freq: 8.0, mean_amplitude: 15.0, peak_amplitude: 45.0,
            spike_rate: 2.5, spectral_centroid: 12.0, spectral_entropy: 0.4,
            burst_index: 0.15, coherence: 0.78,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ImageFeatures {
    pub mycelium_density: f64,
    pub hyphal_width: f64,
    pub branching_frequency: f64,
    pub color_variance: f64,
}

impl Default for ImageFeatures {
    fn default() -> Self {
        Self {
            mycelium_density: 0.65, hyphal_width: 3.2, branching_frequency: 12.0,
            color_variance: 0.3,
        }
    }
}

// ─── Relatório gerado a partir de features + label ─────

#[derive(Clone, Debug)]
pub struct Report {
    pub class_label: String,
    pub confidence: f64,
    pub observations: Vec<String>,
    pub summary: String,
}

pub fn generate_report(
    signal: &SignalFeatures,
    _image: &ImageFeatures,
    class_label: &str,
    confidence: f64,
) -> Report {
    let mut obs: Vec<String> = Vec::new();

    // Observações puramente descritivas, sem atribuir significado
    obs.push(format!(
        "Frequência dominante: {:.1} Hz | Amplitude média: {:.1} µV (pico {:.1} µV)",
        signal.dominant_freq, signal.mean_amplitude, signal.peak_amplitude,
    ));
    obs.push(format!(
        "Spike rate: {:.1}/s | Entropia espectral: {:.3} | Burst index: {:.2}",
        signal.spike_rate, signal.spectral_entropy, signal.burst_index,
    ));
    obs.push(format!(
        "Coerência: {:.2} | Centroide espectral: {:.1} Hz",
        signal.coherence, signal.spectral_centroid,
    ));

    let summary = format!(
        "Sinal classificado como '{}' (confiança {:.1}%) pelo modelo treinado com dados do usuário. \
         Frequência na faixa de {:.0} Hz com amplitude de {:.0} µV, taxa de disparos de {:.1}/s. \
         {}",
        class_label, confidence * 100.0,
        signal.dominant_freq, signal.mean_amplitude, signal.spike_rate,
        if signal.burst_index > 0.3 { "Bursting presente." } else { "Sem bursting significativo." }
    );

    Report { class_label: class_label.to_string(), confidence, observations: obs, summary }
}

// ─── Features sintéticas (demonstração) ──────────────────

pub fn extract_synthetic_features() -> (SignalFeatures, ImageFeatures) {
    use rand::Rng;
    let mut rng = rand::rng();
    let t = rng.random::<f64>();

    let signal = SignalFeatures {
        dominant_freq: 6.0 + t * 20.0,
        mean_amplitude: 5.0 + t * 45.0,
        peak_amplitude: (5.0 + t * 45.0) * (1.5 + rng.random::<f64>() * 2.0),
        spike_rate: 0.5 + t * 8.0,
        spectral_centroid: (6.0 + t * 20.0) * 1.3,
        spectral_entropy: 0.2 + t * 0.6,
        burst_index: 0.05 + t * 0.5,
        coherence: 0.3 + (1.0 - t) * 0.6,
    };

    let image = ImageFeatures {
        mycelium_density: 0.3 + t * 0.6,
        hyphal_width: 2.0 + t * 3.0,
        branching_frequency: 5.0 + t * 20.0,
        color_variance: 0.1 + t * 0.6,
    };

    (signal, image)
}

// ─── Histórico de relatórios ─────────────────────────────

#[derive(Clone)]
pub struct ReportHistory {
    pub entries: Vec<ReportEntry>,
    pub max_entries: usize,
}

#[derive(Clone)]
pub struct ReportEntry {
    pub timestamp: String,
    pub class: String,
    pub confidence: f64,
    pub summary: String,
}

impl Default for ReportHistory {
    fn default() -> Self { Self { entries: Vec::new(), max_entries: 100 } }
}

impl ReportHistory {
    pub fn push(&mut self, ts: String, class: String, conf: f64, summary: String) {
        self.entries.push(ReportEntry { timestamp: ts, class, confidence: conf, summary });
        if self.entries.len() > self.max_entries { self.entries.remove(0); }
    }
}
