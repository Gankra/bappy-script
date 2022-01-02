#[allow(dead_code)]
fn run_typed(input: &str) -> (i64, Option<String>) {
    crate::Program::typed(input).run()
}

// Ideally this is uneeded, but symmetry
