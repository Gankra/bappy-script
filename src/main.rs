use interpretter::*;
use parser::*;

#[cfg(test)]
mod tests;

mod checker;
mod interpretter;
mod parser;

// The program that will be run with `cargo run`
const MAIN_PROGRAM: &str = r#"
    struct Point {
        x: Int,
        y: Int,
    }
    
    let pt = Point { x: 2, y: 7 }
    fn captures(arg: Int) -> Int {
        ret add(arg, pt.y)
    }

    fn no_captures(arg: Int) -> Int {
        ret add(arg, arg)
    }

    fn calls_funcs() -> Int {
        let a = captures(23)
        let b = no_captures(29)
        ret add(a, b)
    }

    let my_fn = captures
    let my_fn2 = no_captures

    print captures(2)
    print my_fn(3)

    print no_captures(6)
    print my_fn2(7)

    print calls_funcs()

    let x = 1
    let y = true
    let z = (x, y)
    if z.1 {
        print add(1, add(x, 2))
    }
    ret 0
"#;

fn main() {
    Program::typed(MAIN_PROGRAM).run();
}

/// Program is kinda a random pile of state that is carried between each
/// phase of the compiler to avoid passing a million arguments to every
/// function. Certain fields (e.g. `main`) will only be populated
/// after certain phases.
///
/// Each phase of the compiler extends this with a function for its phase:
///
/// * parser: `fn parse`
/// * checker: `fn check`
/// * interpretter: `fn eval`
///
/// These 3 functions should be called in sequence. If I was being a try-hard
/// I would create 4 or 5 separate "Program" types that change between each
/// phase to require things be called in order, but I'm not feeling it.
#[derive(Debug, Clone)]
struct Program<'p> {
    /// Should we use static types?
    typed: bool,

    /// The input we're parsing/checking/executing    
    input: &'p str,
    /// Lines of the input that have been parsed
    input_lines: Vec<(usize, &'p str)>,

    /// Fully parsed `main`. Technically "The AST".
    main: Option<Function<'p>>,

    /// Builtin functions ("the stdlib")
    builtins: Vec<Builtin>,

    /// Printed values resulting from `eval` (for testing)
    output: Option<String>,
    /// Hacky state to help the interpretter give better error lines
    cur_eval_span: Span,
}

impl<'p> Program<'p> {
    /// Make a program without static types (may do some runtime type checking).
    #[allow(dead_code)]
    pub fn untyped(input: &'p str) -> Self {
        let mut out = Self::new(input);
        out.typed = false;
        out
    }

    /// Make a program with full static typing (interpretter will still do the checks it does).
    #[allow(dead_code)]
    pub fn typed(input: &'p str) -> Self {
        let mut out = Self::new(input);
        out.typed = true;
        out
    }

    fn new(input: &'p str) -> Self {
        Self {
            typed: true,
            input,
            input_lines: Vec::new(),
            main: None,
            builtins: builtins(),
            output: Some(String::new()),
            cur_eval_span: Span {
                start: addr(input),
                end: addr(input),
            },
        }
    }

    /// Run the program!
    ///
    /// Returns the return value of the implicit `main` and all printed values.
    pub fn run(mut self) -> (i64, Option<String>) {
        println!("parsing...");
        if let Err(e) = self.parse() {
            self.error(
                format!("Parse Error: Unknown Error {:?}", e),
                Span {
                    start: addr(self.input),
                    end: addr(self.input),
                },
            )
        }
        println!("parsed!\n");

        println!("checking...");
        self.check();
        println!("checked!\n");

        println!("evaling...");
        let out = self.eval();
        println!("evaled! {}", out);

        (out, self.output)
    }

    /// Report an error with a proper location in the code.
    #[track_caller]
    pub fn error(&self, message: String, span: Span) -> ! {
        if !self.input_lines.is_empty() {
            let line_number = self.line_number(span);
            let (line_addr, line) = self.input_lines[line_number];
            let start_col = span.start - line_addr;
            let end_col = if span.end <= line_addr + line.len() {
                span.end - line_addr
            } else {
                start_col + 1
            };

            eprintln!("");
            eprintln!("{} @ program.bappy:{}:{}", message, line_number, start_col);
            eprintln!("");
            for i in line_number.saturating_sub(2)..=line_number {
                let (_, line) = self.input_lines[i];
                eprintln!("{:>4} |{}", i, line);
            }
            for _ in 0..start_col + 6 {
                eprint!(" ");
            }
            for _ in start_col..end_col {
                eprint!("~");
            }
            eprintln!();
            eprintln!();
        } else {
            eprintln!("");
            eprintln!("{}", message);
            eprintln!("");
        }

        panic!("{}", message);
    }

    fn line_number(&self, span: Span) -> usize {
        let mut output = 0;
        for (line_number, (addr, _)) in self.input_lines.iter().enumerate() {
            if span.start >= *addr {
                output = line_number
            } else {
                break;
            }
        }
        output
    }
}

/// Little utility for converting substrings of the input to Spans, useful for
/// every phase of the compiler.
pub fn addr(input: &str) -> usize {
    input.as_ptr() as usize
}
