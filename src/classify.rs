// ─── Carregamento e extração de features ───────────────
//
// Lê CSVs, aplica STFT + wavelet denoising, extrai features
// do espectrograma pra classificação.
//
// O pipeline REAL: sinal → STFT → wavelet denoise →
// features espectrais → ridge classifier.
//
// Nada de zero-crossing. Isso é coisa de 2025.
//
// "Frequência é a língua do micélio. A STFT é o dicionário."
//              — Frieren, depois de passar 1000 anos estudando FFT
//
// kyun~ pitch~ (som de FFT sendo computada)

use std::path::Path;
use crate::signal::{self, StftConfig};
use crate::interpret::{SignalFeatures, ImageFeatures, generate_report, Report};

#[derive(Debug, Clone)]
pub struct RawSegment {
    pub start_idx: usize,
    pub end_idx: usize,
    pub time_start: f64,
    pub time_end: f64,
    pub signal: Vec<f64>,
    pub features: SignalFeatures,
    pub ground_truth: String,
    pub image_path: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ClassifiedSegment {
    pub time_start: f64,
    pub time_end: f64,
    pub predicted_class: String,
    pub confidence: f64,
    pub ground_truth: String,
    pub correct: bool,
    pub report: Report,
}

#[derive(Debug, Clone)]
pub struct ClassificationResults {
    pub segments: Vec<ClassifiedSegment>,
    pub accuracy: f64,
    pub confusion: Vec<(String, String, usize)>,
}

// ─── Extração de features via STFT + Wavelet ────────────

pub fn extract_features(signal_data: &[f64]) -> SignalFeatures {
    // Converte pra f32 pro signal module
    let sig_f32: Vec<f32> = signal_data.iter().map(|&x| x as f32).collect();
    let n = sig_f32.len();
    if n < 16 { return SignalFeatures::default(); }

    // 1. Wavelet denoising primeiro
    let denoised = signal::wavelet_denoise(&sig_f32, None, 4);

    // 2. STFT no sinal denoised — window ~1/2 do segmento, n_fft = próximo power-of-two pra boa resolução
    let window_size = (n / 2).min(1024).max(64);
    let config = StftConfig {
        window_size,
        hop_length: (window_size / 4).max(1),
        n_fft: window_size.next_power_of_two().min(2048),
    };

    let (freqs, _times, spec) = signal::stft(&denoised, &config);
    if spec.is_empty() || freqs.is_empty() {
        return SignalFeatures::default();
    }

    let n_freqs = spec.len();
    let n_frames = spec[0].len();

    // 3. Extrai features do espectrograma

    // Energia total por frame
    let mut frame_energy = vec![0.0; n_frames];
    for t in 0..n_frames {
        for f in 0..n_freqs {
            frame_energy[t] += spec[f][t].powi(2);
        }
    }

    // Frequência dominante (centroide ponderado por energia)
    let total_energy: f64 = frame_energy.iter().sum();
    let dominant_freq = if total_energy > 0.0 {
        let mut weighted = 0.0;
        for t in 0..n_frames {
            for f in 0..n_freqs {
                weighted += freqs[f] * spec[f][t].powi(2);
            }
        }
        weighted / total_energy
    } else {
        0.0
    };

    // Amplitude média e pico (do sinal denoised)
    let mean_amp = denoised.iter().map(|&x| x.abs() as f64).sum::<f64>() / n as f64;
    let peak_amp = denoised.iter().map(|&x| x.abs() as f64).fold(0.0f64, f64::max);

    // Spike rate: picos > 3*std no sinal denoised
    let std_dev = (denoised.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
    let spike_thresh = 3.0 * std_dev;
    let spike_count = denoised.iter().filter(|&&x| (x as f64).abs() > spike_thresh).count();
    let spike_rate = spike_count as f64;

    // Centroide espectral
    let spectral_centroid = dominant_freq;

    // Entropia espectral (distribuição de energia entre frequências)
    let spectral_entropy = {
        let mut total_band_energy = 0.0;
        for f in 0..n_freqs {
            for t in 0..n_frames {
                total_band_energy += spec[f][t].powi(2);
            }
        }
        if total_band_energy > 0.0 {
            let mut entropy = 0.0;
            for f in 0..n_freqs {
                let mut band_e = 0.0;
                for t in 0..n_frames {
                    band_e += spec[f][t].powi(2);
                }
                let p = band_e / total_band_energy;
                if p > 0.0 { entropy -= p * p.ln(); }
            }
            (entropy / (n_freqs as f64).ln()).min(1.0)
        } else { 0.0 }
    };

    // Burst index: variância da energia temporal
    let burst = if n_frames > 1 {
        let mean_e = frame_energy.iter().sum::<f64>() / n_frames as f64;
        let var_e = frame_energy.iter().map(|e| (e - mean_e).powi(2)).sum::<f64>() / n_frames as f64;
        (var_e / (mean_e.powi(2) + 1e-10)).min(1.0)
    } else { 0.0 };

    // Coerência: razão entre energia na frequência dominante e total
    let dom_f_idx = if n_freqs > 0 {
        let mut best = 0;
        if dominant_freq > 0.0 {
            let mut min_dist = f64::MAX;
            for f in 0..n_freqs {
                let d = (freqs[f] - dominant_freq).abs();
                if d < min_dist { min_dist = d; best = f; }
            }
        }
        best
    } else { 0 };

    let dom_energy: f64 = (0..n_frames).map(|t| spec[dom_f_idx][t].powi(2)).sum();
    let coherence = if total_energy > 0.0 { (dom_energy / total_energy).min(1.0) } else { 0.0 };

    SignalFeatures {
        dominant_freq, mean_amplitude: mean_amp, peak_amplitude: peak_amp,
        spike_rate, spectral_centroid, spectral_entropy: spectral_entropy,
        burst_index: burst, coherence,
    }
}

/// Vetor de features flat (para o classificador)
pub fn features_to_vec(f: &SignalFeatures) -> Vec<f64> {
    vec![
        f.dominant_freq, f.mean_amplitude, f.peak_amplitude,
        f.spike_rate, f.spectral_centroid, f.spectral_entropy,
        f.burst_index, f.coherence,
    ]
}

// ─── Carrega CSV e retorna segmentos com features ───────

pub fn load_csv(path: &Path) -> anyhow::Result<Vec<RawSegment>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;

    let headers: Vec<String> = rdr.headers()?.iter().map(|h| h.to_string()).collect();
    let time_idx = headers.iter().position(|h| h == "time_ms").unwrap_or(0);
    let sig_idx = headers.iter().position(|h| h.contains("signal_denoised"))
        .or_else(|| headers.iter().position(|h| h.contains("signal_uV")))
        .unwrap_or(1);
    let label_idx = headers.iter().position(|h| h == "label");

    let mut times = Vec::new();
    let mut signal = Vec::new();
    let mut labels: Vec<String> = Vec::new();

    for result in rdr.records() {
        let rec = result?;
        let fields: Vec<&str> = rec.iter().collect();
        if fields.len() <= sig_idx.max(time_idx) { continue; }
        let t: f64 = fields.get(time_idx).unwrap_or(&"0").parse().unwrap_or(0.0);
        let s: f64 = fields.get(sig_idx).unwrap_or(&"0").parse().unwrap_or(0.0);
        times.push(t);
        signal.push(s);
        if let Some(idx) = label_idx {
            labels.push(fields.get(idx).unwrap_or(&"").to_string());
        }
    }

    let has_labels = !labels.is_empty();
    let n = signal.len();

    const SUB_WINDOW: usize = 4000;  // 4s @ 1000Hz
    const SUB_HOP: usize = 2000;     // 50% overlap

    // Segmenta: se tem labels, subsegmenta cada chunk em janelas menores
    let segments = if has_labels {
        let mut segs = Vec::new();
        let mut start = 0usize;
        let mut cur = &labels[0];
        for i in 1..n {
            if labels[i] != *cur || i == n - 1 {
                let end = if i == n - 1 { n } else { i };
                let chunk_signal = &signal[start..end];
                let chunk_tstart = times[start];
                let chunk_tend = times[end - 1];

                // Subsegmenta o chunk labeled em janelas pequenas
                let mut w_start = 0usize;
                while w_start + SUB_WINDOW <= chunk_signal.len() {
                    let w_end = w_start + SUB_WINDOW;
                    let seg_signal = chunk_signal[w_start..w_end].to_vec();
                    let features = extract_features(&seg_signal);
                    let t0 = chunk_tstart + (w_start as f64) / 1000.0;
                    let t1 = chunk_tstart + (w_end as f64) / 1000.0;
                    segs.push(RawSegment {
                        start_idx: start + w_start,
                        end_idx: start + w_end,
                        time_start: t0,
                        time_end: t1,
                        signal: seg_signal,
                        features,
                        ground_truth: cur.clone(),
                        image_path: None,
                    });
                    w_start += SUB_HOP;
                }

                start = i;
                cur = &labels[i];
            }
        }
        segs
    } else {
        let mut segs = Vec::new();
        for (i, chunk) in signal.chunks(SUB_WINDOW).enumerate() {
            if chunk.len() < SUB_WINDOW / 2 { break; }
            let seg_signal = chunk.to_vec();
            let features = extract_features(&seg_signal);
            segs.push(RawSegment {
                start_idx: i * SUB_WINDOW,
                end_idx: (i * SUB_WINDOW + chunk.len()).min(n),
                time_start: times[i * SUB_WINDOW],
                time_end: times[(i * SUB_WINDOW + chunk.len() - 1).min(n - 1)],
                signal: seg_signal, features,
                ground_truth: String::new(),
                image_path: None,
            });
        }
        segs
    };

    Ok(segments)
}

// ─── Carrega experimento (sinal.csv + labels.csv) ──────

pub fn load_experiment(dir: &Path) -> anyhow::Result<Vec<RawSegment>> {
    let sinal_path = dir.join("sinal.csv");
    let labels_path = dir.join("labels.csv");
    if !sinal_path.exists() {
        anyhow::bail!("sinal.csv não encontrado em {}", dir.display());
    }
    if !labels_path.exists() {
        anyhow::bail!("labels.csv não encontrado em {}", dir.display());
    }

    // Lê sinal.csv
    let mut rdr = csv::ReaderBuilder::new().has_headers(true).from_path(&sinal_path)?;
    let mut times = Vec::new();
    let mut signal = Vec::new();
    for result in rdr.records() {
        let rec = result?;
        let t: f64 = rec.get(0).unwrap_or("0").parse().unwrap_or(0.0);
        let s: f64 = rec.get(1).unwrap_or("0").parse().unwrap_or(0.0);
        times.push(t);
        signal.push(s);
    }

    // Lê labels.csv
    let mut rdr = csv::ReaderBuilder::new().has_headers(true).from_path(&labels_path)?;
    let mut segs = Vec::new();

    const SUB_WINDOW: usize = 4000;
    const SUB_HOP: usize = 2000;

    for result in rdr.records() {
        let rec = result?;
        let t_start: f64 = rec.get(0).unwrap_or("0").parse().unwrap_or(0.0);
        let t_end: f64 = rec.get(1).unwrap_or("0").parse().unwrap_or(0.0);
        let classe = rec.get(2).unwrap_or("").to_string();
        let imagem = rec.get(3).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());

        if classe.is_empty() { continue; }

        // Encontra índices no sinal pelo timestamp
        let start_idx = times.iter().position(|&t| t >= t_start).unwrap_or(0);
        let end_idx = times.iter().rposition(|&t| t <= t_end).unwrap_or(signal.len() - 1);
        if end_idx <= start_idx { continue; }

        let chunk_signal = &signal[start_idx..=end_idx];

        let mut w_start = 0usize;
        while w_start + SUB_WINDOW <= chunk_signal.len() {
            let w_end = w_start + SUB_WINDOW;
            let seg_signal = chunk_signal[w_start..w_end].to_vec();
            let features = extract_features(&seg_signal);
            let t0 = t_start + (w_start as f64) / 1000.0;
            let t1 = t_start + (w_end as f64) / 1000.0;
            segs.push(RawSegment {
                start_idx: start_idx + w_start,
                end_idx: start_idx + w_end,
                time_start: t0,
                time_end: t1,
                signal: seg_signal,
                features,
                ground_truth: classe.clone(),
                image_path: imagem.clone(),
            });
            w_start += SUB_HOP;
        }
    }

    Ok(segs)
}

// ─── Aplica classificador treinado nos segmentos ───────

pub fn classify_segments(
    raw: &[RawSegment],
    classifier: &crate::learn::RidgeClassifier,
) -> Vec<ClassifiedSegment> {
    raw.iter().map(|seg| {
        let feat_vec = features_to_vec(&seg.features);
        let (pred, conf) = classifier.predict(&feat_vec);
        let gt = if seg.ground_truth.is_empty() { &pred } else { &seg.ground_truth };
        ClassifiedSegment {
            time_start: seg.time_start,
            time_end: seg.time_end,
            predicted_class: pred.clone(),
            confidence: conf,
            ground_truth: seg.ground_truth.clone(),
            correct: pred == *gt,
            report: generate_report(&seg.features, &ImageFeatures::default(), &pred, conf),
        }
    }).collect()
}

pub fn compute_metrics(segments: &[ClassifiedSegment]) -> ClassificationResults {
    let _total = segments.len();
    let correct = segments.iter().filter(|s| s.correct && !s.ground_truth.is_empty()).count();
    let labeled = segments.iter().filter(|s| !s.ground_truth.is_empty()).count();
    let accuracy = if labeled > 0 { correct as f64 / labeled as f64 } else { 0.0 };

    use std::collections::HashMap;
    let mut confusion: HashMap<(String, String), usize> = HashMap::new();
    for seg in segments {
        if seg.ground_truth.is_empty() { continue; }
        *confusion.entry((seg.predicted_class.clone(), seg.ground_truth.clone())).or_insert(0) += 1;
    }
    let mut conf_vec: Vec<(String, String, usize)> = confusion.into_iter()
        .map(|((p, a), c)| (p, a, c)).collect();
    conf_vec.sort_by(|a, b| b.2.cmp(&a.2));

    ClassificationResults { segments: segments.to_vec(), accuracy, confusion: conf_vec }
}
