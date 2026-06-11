// ─── Aprendizado: classificador treinável pelo usuário ──
//
// Ridge + Random Forest — dois classificadores, um arquivo.
//
// Ridge: (X^T X + λI)^{-1} X^T Y — solução fechada, linear
// RF: Floresta de árvores de decisão com Gini — não-linear
//
// "O Ridge é rápido, o RF é esperto. Um sabe matemática,
//  o outro sabe padrões. Escolha seu lutador."
//
// kyun~ pitch~ (som de floresta crescendo)

use rand::Rng;

// ─── RidgeClassifier ─────────────────────────────────────

#[derive(Clone)]
pub struct RidgeClassifier {
    pub weights: Vec<Vec<f64>>,  // [n_features+1 × n_classes]
    pub classes: Vec<String>,
    pub lambda: f64,
    pub trained: bool,
    mean: Vec<f64>,
    std: Vec<f64>,
}

impl RidgeClassifier {
    pub fn new(lambda: f64) -> Self {
        Self {
            weights: Vec::new(), classes: Vec::new(), lambda, trained: false,
            mean: Vec::new(), std: Vec::new(),
        }
    }

    fn standardize(features: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
        let n = features.len();
        let p = features[0].len();
        let mut mean = vec![0.0; p];
        for f in features {
            for (j, &v) in f.iter().enumerate() {
                mean[j] += v;
            }
        }
        for m in &mut mean { *m /= n as f64; }

        let mut std = vec![0.0; p];
        for f in features {
            for (j, &v) in f.iter().enumerate() {
                let d = v - mean[j];
                std[j] += d * d;
            }
        }
        for s in &mut std {
            *s = ((*s / n as f64).max(1e-12)).sqrt();
        }

        let scaled: Vec<Vec<f64>> = features.iter().map(|f| {
            f.iter().enumerate().map(|(j, &v)| (v - mean[j]) / std[j]).collect()
        }).collect();

        (mean, std, scaled)
    }

    /// Treina com features × labels
    /// features: [n_samples × n_features]
    /// labels: vetor de strings (rótulos fornecidos pelo usuário)
    pub fn fit(&mut self, features: &[Vec<f64>], labels: &[String]) {
        if features.is_empty() || labels.is_empty() { return; }

        let n_samples = features.len();
        let n_features = features[0].len();

        let (mean, std, scaled) = Self::standardize(features);
        self.mean = mean;
        self.std = std;

        // Mapeia labels únicos → índices
        let mut unique: Vec<String> = Vec::new();
        for l in labels {
            if !unique.contains(l) { unique.push(l.clone()); }
        }
        self.classes = unique;
        let n_classes = self.classes.len();

        // Codifica labels como one-hot → matriz alvo Y [n_samples × n_classes]
        let mut y = vec![vec![0.0; n_classes]; n_samples];
        for (i, l) in labels.iter().enumerate() {
            if let Some(j) = self.classes.iter().position(|c| c == l) {
                y[i][j] = 1.0;
            }
        }

        // Monta matriz de design X [n_samples × (n_features + 1)] (bias + features padronizadas)
        let mut x = vec![vec![1.0; n_features + 1]; n_samples];
        for (i, feat) in scaled.iter().enumerate() {
            for (j, &v) in feat.iter().enumerate() {
                x[i][j + 1] = v;
            }
        }

        // Ridge regression: W = (X^T X + λI)^{-1} X^T Y
        let p = n_features + 1;
        let mut xtx = vec![vec![0.0; p]; p];
        let mut xty = vec![vec![0.0; n_classes]; p];

        for i in 0..p {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n_samples {
                    sum += x[k][i] * x[k][j];
                }
                xtx[i][j] = sum;
            }
            if i > 0 { xtx[i][i] += self.lambda * n_samples as f64; }
        }

        for i in 0..p {
            for j in 0..n_classes {
                let mut sum = 0.0;
                for k in 0..n_samples {
                    sum += x[k][i] * y[k][j];
                }
                xty[i][j] = sum;
            }
        }

        self.weights = gaussian_elimination(&xtx, &xty, p, n_classes);
        self.trained = true;
    }

    /// Prediz classe para um vetor de features
    pub fn predict(&self, features: &[f64]) -> (String, f64) {
        if !self.trained || self.classes.is_empty() {
            return ("Desconhecido".into(), 0.0);
        }

        // Padroniza features com mean/std do treino
        let scaled: Vec<f64> = features.iter().enumerate().map(|(j, &v)| {
            (v - self.mean[j]) / self.std[j]
        }).collect();

        // Monta vetor de entrada com bias
        let mut x = vec![1.0; scaled.len() + 1];
        for (i, &v) in scaled.iter().enumerate() {
            x[i + 1] = v;
        }

        // weights[i][j] = coeficiente da feature i para classe j
        // score_j = sum_i x_i * weights[i][j]
        let n_classes = self.classes.len();
        let scores: Vec<f64> = (0..n_classes).map(|j| {
            x.iter().enumerate().map(|(i, &xi)| xi * self.weights[i][j]).sum()
        }).collect();

        // Softmax pra confiança
        let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_s: Vec<f64> = scores.iter().map(|s| (s - max_s).exp()).collect();
        let sum_exp: f64 = exp_s.iter().sum();
        let probs: Vec<f64> = exp_s.iter().map(|e| e / sum_exp).collect();

        // Classe com maior score
        let best_idx = scores.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        (self.classes[best_idx].clone(), probs[best_idx])
    }
}

// ─── Eliminação Gaussiana para sistema linear ───────────

fn gaussian_elimination(a: &[Vec<f64>], b: &[Vec<f64>], n: usize, m: usize) -> Vec<Vec<f64>> {
    // Cria matriz aumentada [A | B]
    let mut aug: Vec<Vec<f64>> = (0..n).map(|i| {
        let mut row = a[i].clone();
        row.extend_from_slice(&b[i]);
        row
    }).collect();

    // Forward elimination
    for col in 0..n {
        // Pivot: encontra linha com maior valor absoluto
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

    // Back substitution
    let mut x = vec![vec![0.0; m]; n];
    for i in (0..n).rev() {
        for j in 0..m {
            let mut sum = aug[i][n + j];
            for k in (i + 1)..n {
                sum -= aug[i][k] * x[k][j];
            }
            if aug[i][i].abs() > 1e-15 {
                x[i][j] = sum / aug[i][i];
            }
        }
    }

    x
}

// ─── Random Forest ────────────────────────────────────────

#[derive(Clone)]
struct TreeNode {
    feature_idx: usize,
    threshold: f64,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
    class_dist: Vec<f64>, // probabilidade por classe se folha
}

#[derive(Clone)]
pub struct RandomForest {
    pub trees: Vec<TreeNode>,
    pub classes: Vec<String>,
    pub n_estimators: usize,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub n_features_subsample: usize,
    pub trained: bool,
}

impl RandomForest {
    pub fn new(n_estimators: usize, max_depth: usize) -> Self {
        Self {
            trees: Vec::new(),
            classes: Vec::new(),
            n_estimators,
            max_depth,
            min_samples_split: 2,
            n_features_subsample: 0,
            trained: false,
        }
    }

    pub fn fit(&mut self, features: &[Vec<f64>], labels: &[String]) {
        if features.is_empty() || labels.is_empty() {
            return;
        }

        let n_samples = features.len();
        let n_features = features[0].len();
        self.n_features_subsample = (n_features as f64).sqrt() as usize;

        let mut unique: Vec<String> = Vec::new();
        for l in labels {
            if !unique.contains(l) {
                unique.push(l.clone());
            }
        }
        self.classes = unique;
        let n_classes = self.classes.len();

        let mut label_ids: Vec<usize> = Vec::new();
        for l in labels {
            let idx = self.classes.iter().position(|c| c == l).unwrap_or(0);
            label_ids.push(idx);
        }

        let mut rng = rand::rng();

        for _ in 0..self.n_estimators {
            // Bootstrap sample
            let mut boot_x: Vec<usize> = Vec::with_capacity(n_samples);
            for _ in 0..n_samples {
                boot_x.push(rng.random_range(0..n_samples));
            }

            let mut tree = TreeNode {
                feature_idx: 0,
                threshold: 0.0,
                left: None,
                right: None,
                class_dist: vec![0.0; n_classes],
            };

            self.build_tree(
                features, &label_ids, &boot_x, n_classes, 0, &mut tree, &mut rng,
            );
            self.trees.push(tree);
        }

        self.trained = true;
    }

    fn build_tree(
        &self,
        all_x: &[Vec<f64>],
        all_y: &[usize],
        indices: &[usize],
        n_classes: usize,
        depth: usize,
        node: &mut TreeNode,
        rng: &mut impl Rng,
    ) {
        let n = indices.len();

        // Conta distribuição de classes
        let mut dist = vec![0.0; n_classes];
        for &idx in indices {
            dist[all_y[idx]] += 1.0;
        }
        let total = n as f64;
        for d in &mut dist {
            *d /= total;
        }
        node.class_dist = dist.clone();

        // Critério de parada
        if depth >= self.max_depth || n <= self.min_samples_split {
            return;
        }

        // Gini atual
        let current_gini = 1.0 - dist.iter().map(|p| p * p).sum::<f64>();
        if current_gini < 1e-12 {
            return; // nó puro
        }

        // Subamostra aleatória de features
        let n_features = all_x[0].len();
        let n_subsample = self.n_features_subsample.min(n_features);
        let mut feat_indices: Vec<usize> = (0..n_features).collect();
        for i in (0..n_features).rev() {
            let j = rng.random_range(0..=i);
            feat_indices.swap(i, j);
        }
        feat_indices.truncate(n_subsample);

        // Busca melhor split
        let mut best_feat = 0;
        let mut best_thresh = 0.0;
        let mut best_gain = 0.0;

        for &feat in &feat_indices {
            let mut values: Vec<f64> = indices.iter().map(|&i| all_x[i][feat]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let n_candidates = (n / 10).max(1).min(20);
            for ti in 0..n_candidates {
                let idx = (ti * n / n_candidates).min(n - 1);
                let thresh = values[idx];

                let (mut left_sum, mut left_n) = (vec![0.0; n_classes], 0.0);
                let (mut right_sum, mut right_n) = (vec![0.0; n_classes], 0.0);

                for &si in indices {
                    if all_x[si][feat] <= thresh {
                        left_sum[all_y[si]] += 1.0;
                        left_n += 1.0;
                    } else {
                        right_sum[all_y[si]] += 1.0;
                        right_n += 1.0;
                    }
                }

                if left_n < 1.0 || right_n < 1.0 {
                    continue;
                }

                let gini_left = 1.0 - left_sum.iter().map(|c| { let d = *c / left_n; d * d }).sum::<f64>();
                let gini_right = 1.0 - right_sum.iter().map(|c| { let d = *c / right_n; d * d }).sum::<f64>();
                let gain = current_gini
                    - (left_n / total) * gini_left
                    - (right_n / total) * gini_right;

                if gain > best_gain {
                    best_gain = gain;
                    best_feat = feat;
                    best_thresh = thresh;
                }
            }
        }

        if best_gain < 1e-12 {
            return; // não achou split útil
        }

        node.feature_idx = best_feat;
        node.threshold = best_thresh;

        let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&si| all_x[si][best_feat] <= best_thresh);

        if left_idx.is_empty() || right_idx.is_empty() {
            return;
        }

        let mut left_node = TreeNode {
            feature_idx: 0,
            threshold: 0.0,
            left: None,
            right: None,
            class_dist: vec![0.0; n_classes],
        };
        self.build_tree(all_x, all_y, &left_idx, n_classes, depth + 1, &mut left_node, rng);
        node.left = Some(Box::new(left_node));

        let mut right_node = TreeNode {
            feature_idx: 0,
            threshold: 0.0,
            left: None,
            right: None,
            class_dist: vec![0.0; n_classes],
        };
        self.build_tree(all_x, all_y, &right_idx, n_classes, depth + 1, &mut right_node, rng);
        node.right = Some(Box::new(right_node));
    }

    fn predict_tree<'a>(node: &'a TreeNode, x: &'a [f64]) -> &'a [f64] {
        match (&node.left, &node.right) {
            (Some(left), Some(right)) => {
                if x[node.feature_idx] <= node.threshold {
                    Self::predict_tree(left, x)
                } else {
                    Self::predict_tree(right, x)
                }
            }
            _ => &node.class_dist,
        }
    }

    pub fn predict(&self, features: &[f64]) -> (String, f64) {
        if !self.trained || self.classes.is_empty() {
            return ("Desconhecido".into(), 0.0);
        }

        let n_classes = self.classes.len();
        let mut votes = vec![0.0; n_classes];

        for tree in &self.trees {
            let dist = Self::predict_tree(tree, features);
            // Hard vote: classe majoritária da folha
            let best = dist
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            votes[best] += 1.0;
        }

        let total = self.trees.len() as f64;
        let best_idx = votes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        (self.classes[best_idx].clone(), votes[best_idx] / total)
    }
}
