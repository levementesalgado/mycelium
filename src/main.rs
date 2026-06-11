// ─── mycelium-net — GUI + CLI ───────────────────────────
//
// Modo normal: GUI interativa
// Modo CLI:    mycelium-net --csv dados/denoised_timecourse.csv
//
// "Se você não entendeu a interface, ela está funcionando."
//                              — Lelouch vi Britannia, UX designer
//
// né~ né~ pitch~ 🍄

mod signal;
mod rc;
mod cnn;
mod ml;
mod interpret;
mod learn;
mod classify;
mod pipeline;
mod rag;
mod gui;

fn main() -> eframe::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 && (args[1] == "--help" || args[1] == "-h") {
        eprintln!("🍄 Mycelium-Net — Classificador de Sinais de Micélio");
        eprintln!();
        eprintln!("USO:");
        eprintln!("  {}                          modo GUI (requer display)", args[0]);
        eprintln!("  {} --csv <caminho>            modo CLI (classifica CSV)", args[0]);
        eprintln!("  {} --csv <caminho> --mode classical|reservoir|cnn|rf", args[0]);
        eprintln!("  {} --csv <caminho> --lambda 0.001 --n 100 --sr 0.9", args[0]);
        eprintln!();
        eprintln!("OPÇÕES:");
        eprintln!("  --csv     <path>   CSV do sinal OU diretório de experimento");
        eprintln!("  --mode    <str>    classical (STFT+Ridge), reservoir (ESN+Ridge), cnn, rf");
        eprintln!("  --lambda  <f64>    regularização Ridge (default: 0.001)");
        eprintln!("  --n       <int>    neurônios do reservoir (default: 100)");
        eprintln!("  --sr      <f64>    raio espectral do reservoir (default: 0.9)");
        eprintln!("  --rag              explicação via LLM (Agnes AI cloud por padrão)");
        eprintln!("  --rag-model <str>   modelo específico (ex: deepseek-r1:7b para Ollama local)");
        eprintln!();
        eprintln!("FORMATOS:");
        eprintln!("  CSV:        time_ms,signal_uV,label (labels opcionais)");
        eprintln!("  Experimento: dir/ com sinal.csv + labels.csv + imagens/");
        eprintln!();
        eprintln!("EXEMPLOS:");
        eprintln!("  {} --csv dados_test/sinais_timecourse.csv", args[0]);
        eprintln!("  {} --csv dados_test/experimento_001", args[0]);
        eprintln!("  {} --csv dados_test/sinais_timecourse.csv --mode reservoir --lambda 0.0001", args[0]);
        eprintln!("  {} --csv dados_test/experimento_001 --rag", args[0]);
        eprintln!("  {} --csv dados_test/experimento_001 --rag --rag-model deepseek-r1:7b   (Ollama local)", args[0]);
        return Ok(());
    }

    // ── Modo CLI ──────────────────────────────────────
    if args.len() > 1 && args[1] == "--csv" {
        let path = match args.get(2) {
            Some(p) => std::path::PathBuf::from(p),
            None => {
                eprintln!("Uso: {} --csv <caminho|dir> [--mode classical|reservoir|cnn|rf] [--lambda 0.001] [--n 100] [--sr 0.9]", args[0]);
                eprintln!("CSVs/disponíveis em: dados/  dados_test/");
                eprintln!("Se for diretório, carrega sinal.csv + labels.csv");
                eprintln!("Modos: classical (STFT+Ridge, default), reservoir (ESN+Ridge), cnn (ESN+CNN), rf (Random Forest)");
                return Ok(());
            }
        };

        let mode = args.iter().position(|a| a == "--mode")
            .and_then(|i| args.get(i + 1))
            .map(|s| match s.as_str() {
                "reservoir" => pipeline::PipelineMode::Reservoir,
                "cnn" => pipeline::PipelineMode::Cnn,
                "rf" | "random_forest" | "randomforest" => pipeline::PipelineMode::RandomForest,
                _ => pipeline::PipelineMode::Classical,
            })
            .unwrap_or(pipeline::PipelineMode::Classical);

        let n_res = args.iter().position(|a| a == "--n")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(100);

        let sr = args.iter().position(|a| a == "--sr")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.9);

        let ridge_lambda = args.iter().position(|a| a == "--lambda")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.001);

        let rag_model = args.iter().position(|a| a == "--rag-model")
            .and_then(|i| args.get(i + 1))
            .map(|s| s.as_str());
        let use_rag = args.iter().any(|a| a == "--rag") || rag_model.is_some();

        let raw_result = if path.is_dir() {
            classify::load_experiment(&path)
        } else {
            classify::load_csv(&path)
        };
        match raw_result {
            Ok(raw_segments) => {
                let mut pipe = match mode {
                    pipeline::PipelineMode::Classical => pipeline::Pipeline::classical(ridge_lambda),
                    pipeline::PipelineMode::Reservoir => pipeline::Pipeline::reservoir(n_res, sr, ridge_lambda),
                    pipeline::PipelineMode::Cnn => pipeline::Pipeline::cnn(n_res, sr, ridge_lambda),
                    pipeline::PipelineMode::RandomForest => pipeline::Pipeline::random_forest(),
                };
                pipe.train(&raw_segments);

                let segments = pipe.classify(&raw_segments);
                let acc = pipe.accuracy(&segments);

                // ── RAG: sumário via LLM ──────────────
                let rag_summary = if use_rag {
                    let client = match rag_model {
                        Some(m) if m == "agnes" || m == "agnes-2.0-flash" => rag::new_agnes(),
                        Some(m) => rag::LlmClient::new_ollama(Some(m)),
                        None => rag::LlmClient::default(),
                    };
                    client.summarize(&segments, 0)
                } else {
                    None
                };

                println!("🍄 Mycelium-Net — Classificacao");
                println!("{}", "─".repeat(60));
                println!("Arquivo: {}", path.display());
                println!("Modo: {}", pipe.summary());
                println!("Segmentos: {} ({} classes)", segments.len(), pipe.classes.len());
                println!("Acurácia: {:.1}%", acc * 100.0);
                println!();
                println!("── Resultados ──");
                for seg in &segments {
                    let mark = if seg.correct && !seg.ground_truth.is_empty() { "✓" } else { " " };
                    println!("  {} [{:.1}s-{:.1}s] pred: {:<20} gt: {:<20} conf: {:.1}%",
                        mark, seg.time_start / 1000.0, seg.time_end / 1000.0,
                        seg.predicted_class, seg.ground_truth,
                        seg.confidence * 100.0);
                }
                if use_rag {
                    println!();
                    match rag_summary {
                        Some(ref text) => println!("🧠 {}", text),
                        None => println!("⚠ LLM: indisponível (ollama rodando?)"),
                    }
                }
            }
            Err(e) => eprintln!("Erro: {}", e),
        }
        return Ok(());
    }

    // ── Modo GUI ──────────────────────────────────────
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("Mycelium-Net — Configurador Nada Intuitivo"),
        ..Default::default()
    };

    eframe::run_native(
        "mycelium-net",
        options,
        Box::new(|_cc| Ok(Box::new(gui::MyceliumGui::new()))),
    )
}
