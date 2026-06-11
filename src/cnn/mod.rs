use candle_core::{Device, Tensor, DType};
use candle_nn::*;

pub struct RcCnnReadout {
    conv1: Conv2d,
    conv2: Conv2d,
    fc: Linear,
    varmap: VarMap,
    pub n_classes: usize,
    pub grid: usize,
}

impl RcCnnReadout {
    pub fn new(n_classes: usize, grid: usize, device: &Device) -> anyhow::Result<Self> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let conv1 = conv2d(1, 32, 3, Conv2dConfig::default(), vs.pp("conv1"))?;
        let conv2 = conv2d(32, 64, 3, Conv2dConfig::default(), vs.pp("conv2"))?;
        let after = grid.saturating_sub(4);
        let flat = 64 * after * after;
        let fc = linear(flat, n_classes, vs.pp("fc"))?;

        Ok(Self { conv1, conv2, fc, varmap, n_classes, grid })
    }

    pub fn forward(&self, input: &Tensor) -> anyhow::Result<Tensor> {
        let x = self.conv1.forward(input)?;
        let x = x.relu()?;
        let x = self.conv2.forward(&x)?;
        let x = x.relu()?;
        let after = self.grid.saturating_sub(4);
        let x = x.reshape(&[1, 64 * after * after])?;
        let x = self.fc.forward(&x)?;
        Ok(x)
    }

    pub fn train(&mut self, states: &[Tensor], targets: &[u32], n_epochs: usize, lr: f64) -> anyhow::Result<()> {
        if states.is_empty() || targets.is_empty() { return Ok(()); }

        let mut opt = AdamW::new_lr(self.varmap.all_vars(), lr)?;
        let n = states.len();

        for _epoch in 0..n_epochs {
            let mut total_loss = 0.0f32;

            for i in 0..n {
                let logits = self.forward(&states[i])?;
                let target = Tensor::new(&[targets[i] as u32], states[i].device())?;
                let loss_val = loss::cross_entropy(&logits, &target)?;

                let grads = loss_val.backward()?;
                opt.step(&grads)?;

                total_loss += loss_val.to_vec0::<f32>()?;
            }

            let avg = total_loss / n as f32;
            if avg < 0.01 { break; }
        }

        Ok(())
    }

    pub fn predict(&self, logits: Tensor) -> anyhow::Result<usize> {
        let data: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
        let best = data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(best)
    }
}
