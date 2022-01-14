use crate::*;

impl<'p> Program<'p> {
    /// Type check the program, including some mild required "compilation".
    ///
    /// The "compilation" is just computing each closure's capture set, which
    /// the runtime needs to know to save the relevant state when it finds a
    /// function decl.
    pub fn run_passes(&mut self) {
        println!("Currently no passes!");
    }
}
