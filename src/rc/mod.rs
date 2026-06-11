use candle_core::{Device, Tensor};

pub struct Reservoir {
    pub weights: Tensor,
    pub input_weights: Tensor,
    pub leak_rate: f64,
}

impl Reservoir {
    /// Cria reservatório com scaling espectral
    ///
    /// Agora o ρ (spectral_radius) é REALMENTE aplicado —
    /// as conexões internas são escaladas pra ter o raio espectral
    /// solicitado. Não é mais placebo.
    pub fn new(n_input: usize, n_reservoir: usize, spectral_radius: f64, device: &Device) -> anyhow::Result<Self> {
        let w_raw = Tensor::randn(0f32, 1.0, (n_reservoir, n_reservoir), device)?;
        let win = Tensor::randn(0f32, 1.0 / n_input as f32, (n_reservoir, n_input), device)?;

        // Estima o raio espectral via power iteration
        let sr = estimate_spectral_radius(&w_raw, 20)?;

        // Escala W pra ter o raio espectral desejado
        let w = if sr > 1e-10 {
            let scale = spectral_radius / sr;
            w_raw.affine(scale, 0.0)?
        } else {
            w_raw
        };

        Ok(Self { weights: w, input_weights: win, leak_rate: 0.3 })
    }

    pub fn step(&self, state: &Tensor, input: &Tensor) -> anyhow::Result<Tensor> {
        let inp_proj = self.input_weights.matmul(input)?;
        let state_proj = self.weights.matmul(state)?;
        let total = (inp_proj + state_proj)?;
        let activation = total.tanh()?;
        let left = state.affine(1.0 - self.leak_rate, 0.0)?;
        let right = activation.affine(self.leak_rate, 0.0)?;
        Ok((left + right)?)
    }

    pub fn run(&self, inputs: &[Tensor]) -> anyhow::Result<Vec<Tensor>> {
        let mut states = Vec::new();
        let n = self.weights.dims()[0];
        let device = inputs[0].device();
        let mut state = Tensor::zeros(&[n], candle_core::DType::F32, device)?;
        for inp in inputs {
            state = self.step(&state, inp)?;
            states.push(state.clone());
        }
        Ok(states)
    }
}

/// Estima o raio espectral via power iteration
fn estimate_spectral_radius(w: &Tensor, n_iter: usize) -> anyhow::Result<f64> {
    let n = w.dims()[0];
    let device = w.device();
    let mut v = Tensor::randn(0f32, 1.0, (n, 1), device)?;
    for _ in 0..n_iter {
        let wv = w.matmul(&v)?;
        let norm = wv.sqr()?.sum_all()?.sqrt()?;
        let flat = norm.flatten_all()?;
        let norm_val: Vec<f32> = flat.to_vec1()?;
        if norm_val.is_empty() || norm_val[0].abs() < 1e-10 { break; }
        v = wv.affine(1.0 / norm_val[0] as f64, 0.0)?;
    }
    let wv = w.matmul(&v)?;
    let v_wv = v.t()?.matmul(&wv)?;
    let flat = v_wv.flatten_all()?;
    let result: Vec<f32> = flat.to_vec1()?;
    Ok(result.first().copied().unwrap_or(0.0).abs() as f64)
}
