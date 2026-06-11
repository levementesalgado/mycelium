use candle_core::{Device, Tensor};
use crate::rc::Reservoir;
use crate::cnn::RcCnnReadout;
use crate::classify::{RawSegment, ClassifiedSegment, features_to_vec, extract_features};
use crate::interpret::{ImageFeatures, generate_report};
use crate::learn::{RidgeClassifier, RandomForest};

#[derive(Clone, Copy, PartialEq)]
pub enum PipelineMode {
    Classical,
    Reservoir,
    Cnn,
    RandomForest,
}

pub struct Pipeline {
    pub mode: PipelineMode,
    pub device: Device,
    pub reservoir: Option<Reservoir>,
    pub n_reservoir: usize,
    pub spectral_radius: f64,
    pub ridge_lambda: f64,
    pub classifier: RidgeClassifier,
    pub rf: RandomForest,
    pub cnn: Option<RcCnnReadout>,
    pub classes: Vec<String>,
    pub trained: bool,
}

impl Pipeline {
    pub fn classical(ridge_lambda: f64) -> Self {
        let device = Device::Cpu;
        Self {
            mode: PipelineMode::Classical,
            device,
            reservoir: None,
            n_reservoir: 100,
            spectral_radius: 0.9,
            ridge_lambda,
            classifier: RidgeClassifier::new(ridge_lambda.max(1e-8)),
            rf: RandomForest::new(50, 6),
            cnn: None,
            classes: Vec::new(),
            trained: false,
        }
    }

    pub fn reservoir(n_reservoir: usize, spectral_radius: f64, ridge_lambda: f64) -> Self {
        let device = Device::Cpu;
        let reservoir = Reservoir::new(1, n_reservoir, spectral_radius, &device);
        let reservoir = match reservoir {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("Erro ao criar reservoir: {}", e);
                None
            }
        };
        Self {
            mode: PipelineMode::Reservoir,
            device,
            reservoir,
            n_reservoir,
            spectral_radius,
            ridge_lambda,
            classifier: RidgeClassifier::new(ridge_lambda),
            rf: RandomForest::new(50, 6),
            cnn: None,
            classes: Vec::new(),
            trained: false,
        }
    }

    pub fn cnn(n_reservoir: usize, spectral_radius: f64, _lr: f64) -> Self {
        let device = Device::Cpu;
        let reservoir = Reservoir::new(1, n_reservoir, spectral_radius, &device);
        let reservoir = match reservoir {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("Erro ao criar reservoir: {}", e);
                None
            }
        };
        Self {
            mode: PipelineMode::Cnn,
            device,
            reservoir,
            n_reservoir,
            spectral_radius,
            ridge_lambda: 1e-3,
            classifier: RidgeClassifier::new(1.0),
            rf: RandomForest::new(50, 6),
            cnn: None,
            classes: Vec::new(),
            trained: false,
        }
    }

    pub fn random_forest() -> Self {
        let device = Device::Cpu;
        Self {
            mode: PipelineMode::RandomForest,
            device,
            reservoir: None,
            n_reservoir: 100,
            spectral_radius: 0.9,
            ridge_lambda: 0.0,
            classifier: RidgeClassifier::new(1.0),
            rf: RandomForest::new(50, 6),
            cnn: None,
            classes: Vec::new(),
            trained: false,
        }
    }

    pub fn extract_features(&self, signal: &[f64]) -> Vec<f64> {
        match self.mode {
            PipelineMode::Classical | PipelineMode::RandomForest => {
                let sf = extract_features(signal);
                features_to_vec(&sf)
            }
            PipelineMode::Reservoir | PipelineMode::Cnn => {
                let sf = extract_features(signal);
                let mut feat = features_to_vec(&sf);
                if let Some(ref res) = self.reservoir {
                    if let Ok(state) = self.run_reservoir(signal, res) {
                        feat.extend(state);
                    }
                }
                feat
            }
        }
    }

    fn run_reservoir(&self, signal: &[f64], reservoir: &Reservoir) -> anyhow::Result<Vec<f64>> {
        let n = signal.len();
        if n < 2 { return Ok(vec![0.0; self.n_reservoir]); }

        let mut inputs = Vec::with_capacity(n);
        for &s in signal {
            let t = Tensor::new(&[s as f32], &self.device)?.reshape(&[1, 1])?;
            inputs.push(t);
        }

        let states = reservoir.run(&inputs)?;
        if states.is_empty() { return Ok(vec![0.0; self.n_reservoir]); }

        let final_state = states.last().unwrap();
        let data: Vec<f32> = final_state.to_vec1()?;
        Ok(data.iter().map(|&x| x as f64).collect())
    }

    pub fn train(&mut self, segments: &[RawSegment]) {
        let labeled: Vec<_> = segments.iter()
            .filter(|s| !s.ground_truth.is_empty()).collect();

        if labeled.is_empty() { return; }

        match self.mode {
            PipelineMode::Classical | PipelineMode::Reservoir => {
                let feats: Vec<Vec<f64>> = labeled.iter()
                    .map(|s| self.extract_features(&s.signal)).collect();
                let labels: Vec<String> = labeled.iter()
                    .map(|s| s.ground_truth.clone()).collect();
                self.classifier.fit(&feats, &labels);
                self.classes = self.classifier.classes.clone();
                self.trained = true;
            }
            PipelineMode::RandomForest => {
                let feats: Vec<Vec<f64>> = labeled.iter()
                    .map(|s| self.extract_features(&s.signal)).collect();
                let labels: Vec<String> = labeled.iter()
                    .map(|s| s.ground_truth.clone()).collect();
                self.rf.fit(&feats, &labels);
                self.classes = self.rf.classes.clone();
                self.trained = true;
            }
            PipelineMode::Cnn => {
                let feats: Vec<Vec<f64>> = labeled.iter()
                    .map(|s| {
                        let sf = extract_features(&s.signal);
                        features_to_vec(&sf)
                    }).collect();
                let labels: Vec<String> = labeled.iter()
                    .map(|s| s.ground_truth.clone()).collect();

                self.classifier.fit(&feats, &labels);
                self.classes = self.classifier.classes.clone();

                if let Some(ref res) = self.reservoir {
                    let grid = (self.n_reservoir as f64).sqrt().ceil() as usize;
                    let n_classes = self.classes.len();
                    if n_classes < 2 { return; }

                    match RcCnnReadout::new(n_classes, grid, &self.device) {
                        Ok(cnn) => {
                            let mut cnn = cnn;
                            let mut states = Vec::new();
                            let mut targets = Vec::new();
                            for seg in labeled.iter() {
                                if let Ok(state) = self.run_reservoir(&seg.signal, res) {
                                    if let Ok(t) = Self::state_to_cnn_input(&state, grid, &self.device) {
                                        states.push(t);
                                        let label_idx = self.classes.iter()
                                            .position(|c| c == &seg.ground_truth).unwrap_or(0);
                                        targets.push(label_idx as u32);
                                    }
                                }
                            }
                            if !states.is_empty() {
                                match cnn.train(&states, &targets, 50, 0.01) {
                                    Ok(_) => {
                                        self.cnn = Some(cnn);
                                        self.trained = true;
                                    }
                                    Err(e) => eprintln!("CNN train error: {}", e),
                                }
                            }
                        }
                        Err(e) => eprintln!("CNN create error: {}", e),
                    }
                }
            }
        }
    }

    pub fn predict(&self, signal: &[f64]) -> (String, f64) {
        match self.mode {
            PipelineMode::Cnn => {
                if let Some(ref cnn) = self.cnn {
                    if let Some(ref res) = self.reservoir {
                        if let Ok(state) = self.run_reservoir(signal, res) {
                            let grid = (self.n_reservoir as f64).sqrt().ceil() as usize;
                            if let Ok(input) = Self::state_to_cnn_input(&state, grid, &self.device) {
                                if let Ok(logits) = cnn.forward(&input) {
                                    if let Ok(pred) = cnn.predict(logits) {
                                        if pred < self.classes.len() {
                                            return (self.classes[pred].clone(), 0.8);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                let feat = self.extract_features(signal);
                self.classifier.predict(&feat)
            }
            PipelineMode::Classical | PipelineMode::Reservoir => {
                let feat = self.extract_features(signal);
                self.classifier.predict(&feat)
            }
            PipelineMode::RandomForest => {
                let feat = self.extract_features(signal);
                self.rf.predict(&feat)
            }
        }
    }

    fn state_to_cnn_input(state: &[f64], grid: usize, device: &Device) -> anyhow::Result<Tensor> {
        let side = grid.max(4);
        let needed = side * side;
        let flat: Vec<f32> = if state.len() >= needed {
            state[..needed].iter().map(|&x| x as f32).collect()
        } else {
            let mut v: Vec<f32> = state.iter().map(|&x| x as f32).collect();
            v.resize(needed, 0.0);
            v
        };
        let t = Tensor::from_slice(&flat, (1, 1, side, side), device)?;
        Ok(t)
    }

    pub fn classify(&self, segments: &[RawSegment]) -> Vec<ClassifiedSegment> {
        segments.iter().map(|seg| {
            let (pred, conf) = self.predict(&seg.signal);
            let feat_signal = extract_features(&seg.signal);
            let gt = if seg.ground_truth.is_empty() { &pred } else { &seg.ground_truth };
            ClassifiedSegment {
                time_start: seg.time_start,
                time_end: seg.time_end,
                predicted_class: pred.clone(),
                confidence: conf,
                ground_truth: seg.ground_truth.clone(),
                correct: pred == *gt,
                report: generate_report(&feat_signal, &ImageFeatures::default(), &pred, conf),
            }
        }).collect()
    }

    pub fn accuracy(&self, segments: &[ClassifiedSegment]) -> f64 {
        let labeled = segments.iter().filter(|s| !s.ground_truth.is_empty()).count();
        let correct = segments.iter().filter(|s| s.correct && !s.ground_truth.is_empty()).count();
        if labeled > 0 { correct as f64 / labeled as f64 } else { 0.0 }
    }

    pub fn summary(&self) -> String {
        match self.mode {
            PipelineMode::Classical => "Clássico (STFT + Ridge)".into(),
            PipelineMode::Reservoir => format!("Reservoir (ESN {} + Ridge)", self.n_reservoir),
            PipelineMode::Cnn => format!("CNN (ESN {} → CNN)", self.n_reservoir),
            PipelineMode::RandomForest => "Random Forest (STFT + RF)".into(),
        }
    }
}
