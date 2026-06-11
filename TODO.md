# mycelium-net — TODO

## 🔴 Alto
- [ ] **Pipeline CNN: n_reservoir precisa ser quadrado perfeito** — state_to_cnn_input agora faz padding/truncamento, mas o ideal é ajustar o reservoir pra gerar grid perfeito
- [ ] **Parâmetros do GUI que não fazem nada** — CnnParams, MlParams, TrainingParams são UI decoration; conectar ao pipeline real
- [ ] **Gaussian elimination duplicada** — copy-paste em `learn.rs` e `ml/mod.rs`; unificar
- [ ] **Dead code** — `ml/mod.rs` inteiro (254 linhas), `classify.rs:classify_segments`, `interpret.rs:ReportHistory`, `signal.rs:ensemble_average`

## 🟡 Médio
- [ ] **Chave Agnes via AGNES_KEY_FILE** — já implementado; documentar variável de ambiente
- [ ] **GUI: Random Forest no ComboBox** — já adicionado; testar se a inferência funciona
- [ ] **GUI: treinar Random Forest** — train_from_labels agora usa `rf.fit()` no modo rf
- [ ] **Zero-pivot na eliminação gaussiana** — `learn.rs:185-199` produz pesos errados quando XᵀX é quase-singular
- [ ] **Pipeline::cnn() ignora learning_rate** — parâmetro `_lr` nunca usado

## 🔵 Baixo
- [ ] **Testes unitários** para extract_features, classify_segments
- [ ] **Documentação dos modos de pipeline** — Classical, Reservoir, CNN, Random Forest
- [ ] **Mistura pt-BR + en** nos comentários — padronizar
- [ ] **Carregar CSV remoto** via URL
- [ ] **Exportar classificação** como CSV com metadados
- [ ] **Benchmark de performance** entre modos de pipeline
