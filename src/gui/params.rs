// ─── Parâmetros do Pipeline (todos expostos em tempo real) ─
//
// Cada struct aqui é um grupo de configuração. Se você não sabe
// o que um parâmetro faz, deixe o default — ou aprenda.
//
// Referência: Kami-tachi ni Hirowareta Otoko — o protagonista
// entende os parâmetros um por um, e no fim domina o sistema.
//
// pitch~ (som do slider sendo arrastado)

#[derive(Clone)]
pub struct RcParams {
    /// Número de neurônios do reservatório (ESN size)
    pub n_reservoir: f64,
    /// Leak rate (0 = memória infinita, 1 = sem inércia)
    pub leak_rate: f64,
    /// Raio espectral (ρ < 1 para estabilidade, > 1 para caos)
    pub spectral_radius: f64,
    /// Escala dos pesos de entrada
    pub input_scale: f64,
    /// Regularização L2 do ridge readout
    pub ridge_lambda: f64,
}

impl Default for RcParams {
    fn default() -> Self {
        Self {
            n_reservoir: 256.0,
            leak_rate: 0.3,
            spectral_radius: 0.9,
            input_scale: 1.0,
            ridge_lambda: 1e-3,
        }
    }
}

#[derive(Clone)]
pub struct SignalParams {
    /// Janela da STFT em samples
    pub window_size: f64,
    /// Hop size da STFT
    pub hop_length: f64,
    /// Pontos da FFT (zero-padded se > window)
    pub n_fft: f64,
    /// Taxa de aquisição em Hz
    pub sample_rate: f64,
}

impl Default for SignalParams {
    fn default() -> Self {
        Self {
            window_size: 256.0,
            hop_length: 128.0,
            n_fft: 256.0,
            sample_rate: 1000.0,
        }
    }
}

#[derive(Clone)]
pub struct CnnParams {
    /// Canais da primeira conv2d
    pub conv1_channels: f64,
    /// Canais da segunda conv2d
    pub conv2_channels: f64,
    /// Tamanho do kernel (3 = 3x3)
    pub kernel_size: f64,
    /// Learning rate do otimizador
    pub learning_rate: f64,
    /// Dropout rate (0 = sem dropout)
    pub dropout: f64,
}

impl Default for CnnParams {
    fn default() -> Self {
        Self {
            conv1_channels: 32.0,
            conv2_channels: 64.0,
            kernel_size: 3.0,
            learning_rate: 1e-3,
            dropout: 0.2,
        }
    }
}

#[derive(Clone)]
pub struct MlParams {
    /// Número de árvores no Random Forest
    pub n_estimators: f64,
    /// Profundidade máxima das árvores
    pub max_depth: f64,
    /// Custo do SVM (kernel radial)
    pub svm_cost: f64,
    /// Gamma do kernel RBF do SVM
    pub svm_gamma: f64,
    /// Threshold do wavelet denoising
    pub wavelet_threshold: f64,
}

impl Default for MlParams {
    fn default() -> Self {
        Self {
            n_estimators: 100.0,
            max_depth: 10.0,
            svm_cost: 1.0,
            svm_gamma: 0.1,
            wavelet_threshold: 0.5,
        }
    }
}

#[derive(Clone)]
pub struct TrainingParams {
    /// Número de épocas
    pub epochs: f64,
    /// Batch size
    pub batch_size: f64,
    /// Fração de validação
    pub val_split: f64,
    /// Patience para early stopping
    pub early_stop_patience: f64,
    /// Regularização L2
    pub l2_reg: f64,
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            epochs: 100.0,
            batch_size: 32.0,
            val_split: 0.2,
            early_stop_patience: 10.0,
            l2_reg: 1e-6,
        }
    }
}

#[derive(Clone)]
pub struct AllParams {
    pub rc: RcParams,
    pub signal: SignalParams,
    pub cnn: CnnParams,
    pub ml: MlParams,
    pub training: TrainingParams,
}

impl Default for AllParams {
    fn default() -> Self {
        Self {
            rc: RcParams::default(),
            signal: SignalParams::default(),
            cnn: CnnParams::default(),
            ml: MlParams::default(),
            training: TrainingParams::default(),
        }
    }
}
