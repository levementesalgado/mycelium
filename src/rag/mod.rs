// ─── RAG: LLM integration for mycelium explanation ─────
//
// Chama LLM (Agnes AI cloud ou Ollama local) pra gerar
// sumário em linguagem natural a partir da classificação.
//
// "O micélio não sabe o que é 'estresse hídrico'.
//  Quem sabe é você. E a LLM só traduz."

use crate::classify::ClassifiedSegment;

const AGNES_BASE: &str = "https://apihub.agnes-ai.com";
const AGNES_MODEL: &str = "agnes-2.0-flash";
const OLLAMA_URL: &str = "http://localhost:11434/api/generate";
const OLLAMA_MODEL: &str = "tinyllama";
const KEY_FILE: &str = "/root/agnes_ai_free_key.md";

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Backend {
    AgnesAi,
    Ollama,
}

#[derive(Debug)]
pub struct LlmClient {
    pub backend: Backend,
    pub model: String,
    pub base_url: String,
    pub api_key: String,
}

impl Default for LlmClient {
    fn default() -> Self {
        let key = read_api_key();
        Self {
            backend: Backend::AgnesAi,
            model: AGNES_MODEL.to_string(),
            base_url: AGNES_BASE.to_string(),
            api_key: key,
        }
    }
}

pub fn new_agnes() -> LlmClient {
    LlmClient::default()
}

impl LlmClient {
    pub fn new_ollama(model: Option<&str>) -> Self {
        Self {
            backend: Backend::Ollama,
            model: model.unwrap_or(OLLAMA_MODEL).to_string(),
            base_url: OLLAMA_URL.to_string(),
            api_key: String::new(),
        }
    }

    pub fn summarize(&self, segments: &[ClassifiedSegment], _top_n: usize) -> Option<String> {
        let sample = segments;

        use std::fmt::Write;
        let mut list = String::new();
        for (i, seg) in sample.iter().enumerate() {
            let _ = writeln!(list, "  {}. [{:.1}s-{:.1}s] Classe: {} (confianca: {:.1}%)",
                i + 1, seg.time_start / 1000.0, seg.time_end / 1000.0,
                seg.predicted_class, seg.confidence * 100.0);
        }

        let prompt = format!(
            "--- CONTEXTO ---\n\
             Este e um experimento de sensoriamento agricola para monitoramento \
             de micelio fungico em solos agricolas. Ainda estamos em fase de \
             experimentacao e validacao do classificador.\n\n\
             --- MISSÃO ---\n\
             Voce e um biologo especialista em micelio fungico aplicado a agricultura.\n\
            Sua tarefa: analisar o experimento abaixo quadro a quadro, \
             como se estivesse olhando um grafico de registro continuo.\n\n\
             --- FORMATO DA RESPOSTA ---\n\
             1. **Linha do tempo**: para cada bloco, explique o estado fisiologico \
             (vegetativo? estresse? esporulacao?) e o que pode ter causado a transicao.\n\
             2. **Biologia por tras**: relacione cada classe com processos reais \
             (ex.: estresse hidrico => reducao de turgor nas hifas; esporulacao => \
             resposta a condicoes adversas; comunicacao micelial => fusao de hifas \
             e troca de nutrientes via anastomose).\n\
             3. **Diagnostico geral**: o micelio esta saudavel? estressado? \
             em qual fase do ciclo de vida? Dê um veredito final.\n\n\
             --- DADOS DO EXPERIMENTO (tempo, classe, confianca) ---\n{}\n",
            list,
        );

        match self.backend {
            Backend::Ollama => {
                let body = serde_json::json!({
                    "model": self.model,
                    "prompt": prompt,
                    "stream": false,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 4096
                    }
                });
                let client = reqwest::blocking::Client::builder()
                    .timeout(std::time::Duration::from_secs(120))
                    .build().ok()?;
                let resp = client.post(&self.base_url).json(&body).send().ok()?;
                let data: serde_json::Value = resp.json().ok()?;
                data["response"].as_str().map(|s| s.trim().to_string())
            }
            Backend::AgnesAi => {
                let body = serde_json::json!({
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "Voce e um biologico especialista em micelio fungico aplicado a sensoriamento agricola. PhD em micologia agricola. Este e um experimento em fase de validacao — seus insights ajudam a calibrar o classificador. Seja detalhado, tecnico e didatico."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 8192,
                    "stream": false
                });
                let url = format!("{}/v1/chat/completions", self.base_url.trim_end_matches('/'));
                let client = reqwest::blocking::Client::builder()
                    .timeout(std::time::Duration::from_secs(300))
                    .build().ok()?;
                let resp = client.post(&url)
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .header("Content-Type", "application/json")
                    .json(&body).send().ok()?;
                let data: serde_json::Value = resp.json().ok()?;
                data["choices"].as_array()
                    .and_then(|arr| arr.first())
                    .and_then(|c| c["message"]["content"].as_str())
                    .map(|s| s.trim().to_string())
            }
        }
    }
}

fn read_api_key() -> String {
    match std::fs::read_to_string(KEY_FILE) {
        Ok(content) => {
            for line in content.lines() {
                if let Some(val) = line.strip_prefix("key: ") {
                    return val.to_string();
                }
            }
            eprintln!("WARN: {} nao contem 'key: '", KEY_FILE);
        }
        Err(e) => eprintln!("WARN: nao foi ler {}: {}", KEY_FILE, e),
    }
    String::new()
}
