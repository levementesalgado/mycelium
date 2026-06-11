// ─── Processamento de Sinais: STFT + Wavelet Denoising ──
//
// STFT: janela de Hann + FFT em cada segmento → espectrograma
// Wavelet: DWT (Daubechies 4) + soft threshold + IDWT
//
// "O micélio fala em frequências. A gente só precisa escutar."
//              — alguém que passou tempo demais com osciloscópio
//
// pitch~ kyun~ (som de FFT sendo computada)

use rustfft::{FftPlanner, num_complex::Complex};

// ─── Coeficientes Daubechies 4 ──────────────────────────
// Filtros de decomposição e reconstrução db4
const DB4_LO_D: [f64; 8] = [
    -0.010597401784997278, 0.032883011666982945, 0.030841381835986965,
    -0.18703481171888114, -0.02798376941698385, 0.6308807679295904,
    0.7148465705525415, 0.23037781330885523,
];
const DB4_HI_D: [f64; 8] = [
    -0.23037781330885523, 0.7148465705525415, -0.6308807679295904,
    -0.02798376941698385, 0.18703481171888114, 0.030841381835986965,
    -0.032883011666982945, -0.010597401784997278,
];
const DB4_LO_R: [f64; 8] = [
    0.23037781330885523, 0.7148465705525415, 0.6308807679295904,
    -0.02798376941698385, -0.18703481171888114, 0.030841381835986965,
    0.032883011666982945, -0.010597401784997278,
];
const DB4_HI_R: [f64; 8] = [
    -0.010597401784997278, -0.032883011666982945, 0.030841381835986965,
    0.18703481171888114, -0.02798376941698385, -0.6308807679295904,
    0.7148465705525415, -0.23037781330885523,
];

// ─── STFT ───────────────────────────────────────────────

pub struct StftConfig {
    pub window_size: usize,
    pub hop_length: usize,
    pub n_fft: usize,
}

impl Default for StftConfig {
    fn default() -> Self {
        Self { window_size: 256, hop_length: 128, n_fft: 256 }
    }
}

/// Retorna: (frequencias, tempos, espectrograma) onde espectrograma[f][t] = magnitude
pub fn stft(signal: &[f32], config: &StftConfig) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let n = signal.len();
    if n < config.window_size {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    // Pré-calcula janela de Hann
    let window: Vec<f64> = (0..config.window_size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (config.window_size as f64 - 1.0)).cos()))
        .collect();

    // Planeja FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(config.n_fft);

    let n_freqs = config.n_fft / 2 + 1;
    let n_frames = 1 + (n - config.window_size) / config.hop_length;

    let mut spectrogram: Vec<Vec<f64>> = vec![vec![0.0; n_frames]; n_freqs];
    let mut times = vec![0.0; n_frames];
    let freqs: Vec<f64> = (0..n_freqs).map(|i| i as f64).collect();

    for frame in 0..n_frames {
        let start = frame * config.hop_length;
        let mut fft_input: Vec<Complex<f64>> = (0..config.n_fft)
            .map(|i| {
                if i < config.window_size {
                    let idx = start + i;
                    if idx < n {
                        Complex::new(signal[idx] as f64 * window[i], 0.0)
                    } else {
                        Complex::new(0.0, 0.0)
                    }
                } else {
                    Complex::new(0.0, 0.0)
                }
            })
            .collect();

        fft.process(&mut fft_input);

        // Magnitude
        for f in 0..n_freqs {
            spectrogram[f][frame] = fft_input[f].norm() / config.n_fft as f64;
        }

        times[frame] = start as f64;
    }

    (freqs, times, spectrogram)
}

// ─── Wavelet Denoising ─────────────────────────────────

/// Aplica convolução 1D com padding
fn convolve(signal: &[f64], filter: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let m = filter.len();
    let mut result = vec![0.0; n + m - 1];
    for i in 0..n {
        for j in 0..m {
            result[i + j] += signal[i] * filter[j];
        }
    }
    result
}

/// Downsample por 2
fn downsample(signal: &[f64]) -> Vec<f64> {
    signal.iter().step_by(2).copied().collect()
}

/// Upsample (insere zero entre amostras)
fn upsample(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let mut result = vec![0.0; n * 2];
    for i in 0..n {
        result[i * 2] = signal[i];
    }
    result
}

/// DWT de um nível: retorna (aproximação, detalhe)
fn dwt_one_level(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let lo = convolve(signal, &DB4_LO_D);
    let hi = convolve(signal, &DB4_HI_D);
    let approx = downsample(&lo);
    let detail = downsample(&hi);
    (approx, detail)
}

/// IDWT de um nível: reconstrói a partir de (aproximação, detalhe)
fn idwt_one_level(approx: &[f64], detail: &[f64]) -> Vec<f64> {
    let up_a = upsample(approx);
    let up_d = upsample(detail);
    let rec_a = convolve(&up_a, &DB4_LO_R);
    let rec_d = convolve(&up_d, &DB4_HI_R);

    // Soma e trunca ao tamanho esperado (2 * len(approx))
    let expected = approx.len() * 2;
    let mut result = vec![0.0; expected];
    for i in 0..expected {
        let mut v = 0.0;
        if i < rec_a.len() { v += rec_a[i]; }
        if i < rec_d.len() { v += rec_d[i]; }
        result[i] = v;
    }
    result
}

/// Wavelet denoising completo: DWT multinível → soft threshold → IDWT
///
/// - signal: sinal de entrada
/// - threshold: limiar de soft thresholding (None = universal Donoho-Johnstone)
/// - level: número de níveis de decomposição
pub fn wavelet_denoise(signal: &[f32], threshold: Option<f64>, level: usize) -> Vec<f32> {
    let n = signal.len();
    if n < 4 { return signal.to_vec(); }

    let sig: Vec<f64> = signal.iter().map(|&x| x as f64).collect();

    // Decomposição: armazena coeficientes de detalhe por nível
    let mut details: Vec<Vec<f64>> = Vec::with_capacity(level);
    let mut current = sig;

    for _ in 0..level {
        if current.len() < 4 { break; }
        let (approx, detail) = dwt_one_level(&current);
        details.push(detail);
        current = approx;
    }

    // Threshold universal (Donoho-Johnstone)
    let thr = threshold.unwrap_or_else(|| {
        if let Some(last) = details.last() {
            let n_s = last.len() as f64;
            let median = {
                let mut sorted = last.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                sorted[sorted.len() / 2]
            };
            let sigma = median.abs() / 0.6745; // MAD estimator
            sigma * (2.0 * n_s.ln()).sqrt()
        } else {
            0.1
        }
    });

    // Soft threshold em todos os coeficientes de detalhe
    for d in &mut details {
        for v in d.iter_mut() {
            if v.abs() <= thr {
                *v = 0.0;
            } else {
                *v = v.signum() * (v.abs() - thr);
            }
        }
    }

    // Reconstrução
    let mut result = current;
    for d in details.into_iter().rev() {
        // Ajusta tamanhos para IDWT (precisa ser par)
        let max_len = result.len().max(d.len());
        let mut a = result;
        let mut det = d;
        if a.len() < max_len { a.push(0.0); }
        if det.len() < max_len { det.push(0.0); }
        if a.len() > max_len { a.truncate(max_len); }
        if det.len() > max_len { det.truncate(max_len); }

        result = idwt_one_level(&a, &det);
    }

    // Trunca ao tamanho original
    result.truncate(n);
    result.iter().map(|&x| x as f32).collect()
}

// ─── Ensemble Averaging ─────────────────────────────────

pub fn ensemble_average(epochs: &[Vec<f32>]) -> Vec<f32> {
    if epochs.is_empty() { return Vec::new(); }
    let n = epochs[0].len();
    let mut sum = vec![0.0f32; n];
    for epoch in epochs {
        for (i, &v) in epoch.iter().enumerate().take(n) {
            sum[i] += v;
        }
    }
    for v in &mut sum {
        *v /= epochs.len() as f32;
    }
    sum
}
