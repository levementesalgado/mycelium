# mycelium-net 🍄

**Computação neuromórfica com redes miceliais — pesquisa to.**

Pipeline Rust + R para aquisição, processamento e classificação de sinais elétricos
de fungos (micélio), usando Reservoir Computing + CNN como arquitetura de readout.
Aplicação alvo: agricultura de precisão (detecção de estresse hídrico/nutricional via
fungos micorrízicos arbusculares — AMF).

## Rápido

```bash
# CLI — classificar sinais sintéticos
cargo run --release -- --csv dados_test/sinais_timecourse.csv

# GUI — interface interativa
cargo run --release

# R — gerar sinais sintéticos + imagens STFT/wavelet
Rscript R/generate_signals.R
```

## Arquitetura

```
Micélio (fungo)
   │ sinais elétricos (μV)
   ▼
Eletrodos IrOx → INA118 (6000×) → ADC 16-bit → USB
   │ série temporal crua
   ▼
┌──────────────────────────────────┐
│  PROCESSAMENTO (Rust + R)        │
│                                  │
│  wavelet_denoise(db4, soft_thr)  │
│  STFT (Hann window + FFT)        │
│  → 8 features espectrais         │
└──────────────┬───────────────────┘
               │ feature vector
               ▼
┌──────────────────────────────────┐
│  CLASSIFICAÇÃO (Rust)            │
│                                  │
│  RidgeClassifier (treinado pelo  │
│  usuário via GUI ou CLI)         │
│  → classe + confiança (softmax)  │
└──────────────────────────────────┘
```

Componentes **não integrados** (implementados mas não conectados ao pipeline):
- **Reservoir (ESN)**: `src/rc/mod.rs` — leaky integrator, sem scaling espectral
- **CNN readout**: `src/cnn/mod.rs` — forward Conv2d(1→32→64)+Linear, sem treinamento
- **GradientBoosting**: `src/ml/mod.rs` — stump-based regressor, sem wrapper de classificação

## Estrutura

```
mycelium-net/
├── src/
│   ├── main.rs          # CLI (--csv) + GUI (eframe)
│   ├── signal/mod.rs    # STFT + wavelet denoising (rustfft, db4 hand-rolled)
│   ├── classify.rs      # Feature extraction (8 features) + segmentação CSV
│   ├── learn.rs         # RidgeClassifier — ridge regression + softmax
│   ├── ml/mod.rs        # RcReadoutRidge + GradientBoosting (standalone)
│   ├── rc/mod.rs        # Echo State Network (leaky integrator, candle)
│   ├── cnn/mod.rs       # CNN readout forward pass (candle-nn)
│   ├── interpret/mod.rs # Report generator (descritivo, sem classes hardcoded)
│   └── gui/
│       ├── mod.rs       # GUI principal (egui 0.34, CentralPanel único)
│       ├── params.rs    # AllParams: RC, Signal, CNN, ML, Training
│       ├── pipeline.rs  # Diagrama visual do pipeline (5 estágios)
│       └── widgets.rs   # Knob circular (não usado — reservado)
├── R/
│   ├── generate_signals.R     # Gera 8 classes sintéticas + PNGs STFT/wavelet
│   ├── process_experiment.R   # Processa CSV real → PNGs + clean CSV + relatório
│   ├── signal_analysis.R      # R&D: STFT, wavelet, ensemble averaging, RF, ARIMA
│   └── ml_baselines.R         # SVM + RandomForest (baseline para paper)
├── dados/               # Sinais sintéticos (gerados por generate_signals.R)
├── dados_test/          # Cópia de dados/ para testes na GUI
├── dados_real/          # Coloque CSVs reais aqui (vazio inicialmente)
└── Cargo.toml           # Dependências: candle-core, eframe, rustfft, ...
```

## Pré-requisitos

### Rust (≥ 1.85)
```bash
cargo build --release
```

Dependências principais: `candle-core`/`candle-nn` para tensores, `eframe`/`egui` para GUI,
`rustfft` para STFT, `csv` para parsing, `image` para carregar PNGs.

### R (≥ 4.0) — para gerar sinais sintéticos e processar experimentos
```bash
install.packages(c("signal", "wavelets", "ggplot2", "forecast", "tseries",
                   "randomForest", "e1071", "imager"))
```

## Uso

### CLI — classificação em lote
```bash
cargo run --release -- --csv dados_test/sinais_timecourse.csv
```

Carrega o CSV, extrai features (STFT + wavelet), treina RidgeClassifier nos segmentos
com ground truth, e imprime acurácia + matriz de confusão.

### GUI — exploração interativa
```bash
cargo run --release
```

Janela única com `CentralPanel`:
- Barra superior: título, tema (🌙/☀️), fps, CSV carregado
- Barra de ações: 📂 Teste (dados_test/), 📂 Real (dados_real/), 🖼 Imagens, 🏋 Treinar, 🔄 Reset
- Navegação horizontal do pipeline (5 estágios — visual, não funcional)
- Coluna esquerda: plot do sinal, visualização de imagens, rotulagem de segmentos
- Coluna direita: diagrama do pipeline, resultados da classificação, parâmetros (toggle ⚙)
- Parâmetros organizados em grupos: Reservoir, STFT/Wavelet, CNN, ML, Training

### R — geração de sinais sintéticos
```bash
Rscript R/generate_signals.R
```

Gera 8 classes: Vegetativo, EstresseHídrico, EstresseNutricional, RespostaLesão,
ComunicaçãoMicelial, Esporulação, Senescência, RuídoInespecífico.
Produz CSVs + PNGs (signal overlay, spectrograma STFT, coeficientes wavelet).

### R — processamento de experimentos reais
```bash
Rscript R/process_experiment.R dados_real/seu_experimento.csv
# ou batch:
Rscript R/process_experiment.R --dir dados_real/
```

## Pipeline de processamento (classify.rs)

1. **Segmentação**: divide o CSV em segmentos (~1s cada ou por transição de label)
2. **Wavelet denoising**: DWT Daubechies 4 (multinível) + soft threshold
   (Donoho–Johnstone universal) + IDWT
3. **STFT**: Hann window + FFT (rustfft) → magnitude spectrograma
4. **Features** (8 dimensões):
   - `dominant_freq` — frequência de pico no espectro
   - `mean_amplitude` — amplitude média do sinal denoised
   - `peak_amplitude` — amplitude máxima
   - `spike_rate` — taxa de spikes (eventos > 3σ)
   - `spectral_centroid` — centro de massa espectral
   - `spectral_entropy` — entropia normalizada do espectro
   - `burst_index` — razão entre amplitude em bursting vs silent
   - `coherence` — coesão espectral (largura do pico dominante)
5. **Classificação**: Ridge regression (one-hot encoding + fechado via eliminação Gaussiana)
   + softmax para confiança

## Parâmetros (gui/params.rs)

| Grupo | Parâmetro | Default | Descrição |
|-------|-----------|---------|-----------|
| Reservoir | N | 256 | Neurônios do ESN |
| | leak | 0.3 | Leak rate (0 = memória infinita) |
| | ρ | 0.9 | Raio espectral (⚠ não aplicado no código atual) |
| | in_scale | 1.0 | Escala dos pesos de entrada |
| | λ | 1e-3 | Ridge regularization |
| STFT | window | 256 | Tamanho da janela (samples) |
| | hop | 128 | Hop size |
| | n_fft | 256 | Pontos FFT (zero-padded) |
| | fs | 1000 | Taxa de aquisição (Hz) |
| CNN | C1 | 32 | Canais primeira convolução |
| | C2 | 64 | Canais segunda convolução |
| | K | 3 | Kernel size |
| | lr | 1e-3 | Learning rate |
| | dropout | 0.2 | Dropout rate |
| ML | trees | 100 | Random Forest (R apenas) |
| | depth | 10 | Max depth |
| | C | 1.0 | SVM cost (R apenas) |
| | γ | 0.1 | SVM gamma (R apenas) |
| | w_thresh | 0.5 | Wavelet threshold multiplier |
| Training | epochs | 100 | Épocas |
| | batch | 32 | Batch size |
| | val% | 0.2 | Val split |
| | patience | 10 | Early stopping |
| | L2 | 1e-6 | L2 regularization |

## O que funciona / não funciona

### ✅ Funciona
- **CLI**: `--csv` carrega CSV, extrai features, treina RidgeClassifier, acurácia + confusão
- **GUI**: carregar CSVs, visualizar sinais, rotular segmentos, treinar modelo, ver resultados
- **GUI**: visualizar imagens PNG (STFT, wavelet, denoised)
- **GUI**: alternar tema (🌙/☀️) — toda a janela respeita o tema
- **GUI**: parâmetros ajustáveis em tempo real (toggle ⚙ Params)
- **STFT**: magnitude spectrograma com Hann window
- **Wavelet denoising**: DWT db4 multinível + soft threshold + IDWT
- **Ridge Classifier**: treinamento fechado (Gaussian elimination) + softmax
- **R**: geração de 8 classes sintéticas com perfil bioelétrico realista
- **R**: processamento de CSV experimental → PNGs + clean CSV + relatório
- **R**: baselines SVM e RandomForest para comparação

### ⚠️ Implementado mas não integrado
- **ESN (Reservoir)**: `src/rc/` compila, mas `spectral_radius` não é escalado nas
  matrizes de peso; não conectado ao pipeline de classificação
- **CNN readout**: `src/cnn/` tem forward pass mas sem treinamento (backprop);
  requer integração com reservoir states
- **GradientBoosting**: `src/ml/` tem stump-based regressor mas não é usado
  pelo pipeline (não há wrapper de classificação)
- **Pipeline diagram**: 5 estágios clicáveis na GUI mas puramente visuais
  (não alternam modo de processamento)

### ❌ Não implementado
- Aquisição serial (Rust serialport)
- Integração RC → CNN end-to-end
- Logging HDF5
- Validação cruzada temporal
- Testes unitários
- Image feature extraction (ImageFeatures existe mas não extrai de imagens reais)

## Protocolo experimental (3 fases)

### Fase 1 — Validação da cadeia (30 dias)
Shiitake + eletrodos IrOx + INA118 → detectar atividade elétrica
(LaRocco et al. 2025).

### Fase 2 — Detecção em rizosfera (60 dias)
Vegetação nativa APP Fatec com AMF → mapeamento + medições campo.

### Fase 3 — Hipótese AMF em soja/milho (90 dias)
*Glycine max* + *Rhizophagus irregularis* / *Funneliformis mosseae*
com déficit hídrico como estressor.

## Referências

- LaRocco et al. (2025) — Memristores de shiitake, PLOS One
- Telhan et al. (2025) — Chips micélio + PEDOT:PSS, bioRxiv
- Tompris et al. (2025) — Modelagem crescimento micelial, Nat. Comp.
- Dell'Aversana (2025) — Mycelial_Net classificação mineral, Minerals
- Buffi et al. (2025) — Desafios eletrofisiologia fúngica, FEMS
- Adamatzky (2018) — Spiking behaviour *Pleurotus djamor*, Sci. Rep.
