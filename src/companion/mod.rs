pub struct Companion {
    pub name: String,
    pub messages: Vec<(String, String)>, // (role, content)
}

impl Companion {
    pub fn new() -> Self {
        Self {
            name: "Chibi".into(),
            messages: Vec::new(),
        }
    }
}
