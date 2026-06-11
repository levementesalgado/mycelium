// ─── Modelos ML: Ridge Readout + Gradient Boosting ────
//
// Ridge: (X^T X + λI)^{-1} X^T Y — fechado, direto, sem graça
// Gradient Boosting: floresta de stumps que aprende resíduos
//
// "Um modelo simples que funciona > um modelo complexo que
//  não converge." — Shiro (No Game No Life, ep. 6)
//
// pitch~ (som de gradiente descendente)

// ─── Ridge Readout (f64 puro, sem candle) ──────────────

pub struct RcReadoutRidge {
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub lambda: f64,
    pub input_size: usize,
    pub output_size: usize,
    pub trained: bool,
}

impl RcReadoutRidge {
    pub fn new(input_size: usize, output_size: usize, lambda: f64) -> Self {
        Self {
            weights: Vec::new(), bias: Vec::new(),
            lambda, input_size, output_size, trained: false,
        }
    }

    /// Treina: X[n×d], Y[n×c] → W[d×c]
    /// Usa matriz aumentada [1 X] pra incluir bias
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[Vec<f64>]) {
        let n = x.len();
        if n == 0 { return; }
        let d = self.input_size;
        let c = self.output_size;
        let p = d + 1; // +1 pro bias

        // Matriz aumentada X_aug = [1 X]
        let mut x_aug = vec![vec![1.0; p]; n];
        for i in 0..n {
            for j in 0..d {
                x_aug[i][j + 1] = x[i][j];
            }
        }

        // X^T X
        let mut xtx = vec![vec![0.0; p]; p];
        let mut xty = vec![vec![0.0; c]; p];

        for i in 0..p {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n { sum += x_aug[k][i] * x_aug[k][j]; }
                xtx[i][j] = sum;
            }
            if i > 0 { xtx[i][i] += self.lambda * n as f64; }

            for j in 0..c {
                let mut sum = 0.0;
                for k in 0..n { sum += x_aug[k][i] * y[k][j]; }
                xty[i][j] = sum;
            }
        }

        // Resolve por eliminação gaussiana
        let w = gaussian_elimination(&xtx, &xty, p, c);

        // Separa bias do resto
        self.bias = w[0].clone();
        self.weights = w[1..].to_vec();
        self.trained = true;
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = x.len();
        let c = self.output_size;
        let mut preds = vec![vec![0.0; c]; n];

        if !self.trained { return preds; }

        for i in 0..n {
            for j in 0..c {
                let mut sum = self.bias[j];
                for k in 0..self.input_size {
                    sum += x[i][k] * self.weights[k][j];
                }
                preds[i][j] = sum;
            }
        }

        preds
    }

    pub fn predict_class(&self, x: &[f64]) -> (usize, f64) {
        if !self.trained { return (0, 0.0); }
        let mut scores = vec![0.0; self.output_size];
        for j in 0..self.output_size {
            let mut sum = self.bias[j];
            for k in 0..self.input_size {
                sum += x[k] * self.weights[k][j];
            }
            scores[j] = sum;
        }
        let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_s: Vec<f64> = scores.iter().map(|s| (s - max_s).exp()).collect();
        let sum_exp: f64 = exp_s.iter().sum();
        let best = scores.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        (best, exp_s[best] / sum_exp)
    }
}

// ─── Gradient Boosting (Regressor) ──────────────────────

pub struct GradientBoosting {
    pub n_estimators: usize,
    pub learning_rate: f64,
    trees: Vec<Stump>,
    base_pred: f64,
    n_features: usize,
}

struct Stump {
    threshold: f64,
    feature_idx: usize,
    left_val: f64,
    right_val: f64,
}

impl GradientBoosting {
    pub fn new(n_estimators: usize, learning_rate: f64) -> Self {
        Self { n_estimators, learning_rate, trees: Vec::new(), base_pred: 0.0, n_features: 0 }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        let n = x.len();
        if n == 0 { return; }
        self.n_features = x[0].len();

        // Predição inicial = média
        self.base_pred = y.iter().sum::<f64>() / n as f64;
        let mut residuals: Vec<f64> = y.iter().map(|&yi| yi - self.base_pred).collect();
        let mut prev_loss = residuals.iter().map(|r| r.powi(2)).sum::<f64>();

        for _ in 0..self.n_estimators {
            let stump = self.find_best_stump(x, &residuals);
            for i in 0..n {
                let pred = if x[i][stump.feature_idx] <= stump.threshold { stump.left_val } else { stump.right_val };
                residuals[i] -= self.learning_rate * pred;
            }
            self.trees.push(stump);

            let loss = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
            if loss > prev_loss * 0.99 || loss < 1e-12 { break; }
            prev_loss = loss;
        }
    }

    fn find_best_stump(&self, x: &[Vec<f64>], residuals: &[f64]) -> Stump {
        let n = x.len();
        let mut best = Stump { threshold: 0.0, feature_idx: 0, left_val: 0.0, right_val: 0.0 };
        let mut best_loss = f64::MAX;

        for feat in 0..self.n_features {
            let mut values: Vec<f64> = x.iter().map(|row| row[feat]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let n_thr = (n / 10).max(1).min(20);
            for ti in 1..n_thr {
                let idx = (ti * n / n_thr).min(n - 1);
                let thresh = values[idx];

                let (mut ls, mut ln) = (0.0, 0.0);
                let (mut rs, mut rn) = (0.0, 0.0);
                for i in 0..n {
                    if x[i][feat] <= thresh { ls += residuals[i]; ln += 1.0; }
                    else { rs += residuals[i]; rn += 1.0; }
                }
                let lv = if ln > 0.0 { ls / ln } else { 0.0 };
                let rv = if rn > 0.0 { rs / rn } else { 0.0 };

                let mut loss = 0.0;
                for i in 0..n {
                    let pred = if x[i][feat] <= thresh { lv } else { rv };
                    loss += (residuals[i] - pred).powi(2);
                }
                if loss < best_loss {
                    best_loss = loss;
                    best = Stump { threshold: thresh, feature_idx: feat, left_val: lv, right_val: rv };
                }
            }
        }
        best
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let n = x.len();
        let mut preds = vec![self.base_pred; n];
        for stump in &self.trees {
            for i in 0..n {
                let delta = if x[i][stump.feature_idx] <= stump.threshold { stump.left_val } else { stump.right_val };
                preds[i] += self.learning_rate * delta;
            }
        }
        preds
    }

    pub fn predict_single(&self, x: &[f64]) -> f64 {
        let mut pred = self.base_pred;
        for stump in &self.trees {
            pred += self.learning_rate * if x[stump.feature_idx] <= stump.threshold { stump.left_val } else { stump.right_val };
        }
        pred
    }
}

// ─── Eliminação Gaussiana ──────────────────────────────

fn gaussian_elimination(a: &[Vec<f64>], b: &[Vec<f64>], n: usize, m: usize) -> Vec<Vec<f64>> {
    let mut aug: Vec<Vec<f64>> = (0..n).map(|i| {
        let mut row = a[i].clone();
        row.extend_from_slice(&b[i]);
        row
    }).collect();

    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| {
            aug[i][col].abs().partial_cmp(&aug[j][col].abs()).unwrap()
        }).unwrap_or(col);
        if aug[pivot][col].abs() < 1e-15 { continue; }
        aug.swap(col, pivot);
        for row in (col + 1)..n {
            let factor = aug[row][col] / aug[col][col];
            for k in col..(n + m) {
                aug[row][k] -= factor * aug[col][k];
            }
        }
    }

    let mut x = vec![vec![0.0; m]; n];
    for i in (0..n).rev() {
        for j in 0..m {
            let mut sum = aug[i][n + j];
            for k in (i + 1)..n {
                sum -= aug[i][k] * x[k][j];
            }
            if aug[i][i].abs() > 1e-15 { x[i][j] = sum / aug[i][i]; }
        }
    }
    x
}
