#![allow(clippy::too_many_arguments)]
#![allow(clippy::for_kv_map)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::enum_variant_names)]
#![allow(dead_code)]
#![allow(clippy::only_used_in_recursion)]

use checker::*;
use interpretter_ast::*;
use interpretter_cfg::*;
use parser::*;

#[cfg(test)]
mod tests;

mod checker;
mod interpretter_ast;
mod interpretter_cfg;
mod parser;
mod passes;

// The program that will be run with `cargo run`
const MAIN_PROGRAM: &str = r#"
let factor: Int = 3
fn get_factor() -> Int {
    ret factor
}
print get_factor()
fn multi(factory: fn() -> Int, x: Int) -> Int {
    print x
    print factory()
    ret mul(x, factory())
}

let x: Int = 7

print multi(get_factor, x)  


fn mega_multi(multiplier: fn(fn() -> Int, Int) -> Int) -> () {
    fn eleven() -> Int {
        ret 11
    }
    print multiplier(eleven, 9)
    ret ()
}

let _: () = mega_multi(multi)

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
#[derive(Debug)]
struct Program<'p> {
    /// Should we use static types?
    typed: bool,

    /// The input we're parsing/checking/executing    
    input: &'p str,
    /// Lines of the input that have been parsed
    input_lines: Vec<(usize, &'p str)>,

    /// Fully parsed `main`. Technically "The AST".
    ast_main: Option<Function<'p>>,

    /// Fully constructed typing context
    ctx: Option<TyCtx<'p>>,
    /// Fully constructed SSA CFG (control flow graph)
    cfg: Option<Cfg<'p>>,

    /// Builtin functions ("the stdlib")
    builtins: Vec<Builtin>,

    /// Printed values resulting from `eval` (for testing)
    output: Option<String>,
    /// Hacky state to help the interpretter give better error lines
    cur_eval_span: Span,
}

#[derive(Clone)]
pub struct Builtin {
    pub name: &'static str,
    pub args: &'static [&'static str],
    pub ty: TyName<'static>,
    pub layout: FrameLayout,
    pub ast_impl: for<'e, 'p> fn(args: &[Val<'e, 'p>]) -> Val<'e, 'p>,
    pub cfg_impl: fn(&mut CfgInterpretter) -> (),
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
            ast_main: None,
            cfg: None,
            ctx: None,
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
        println!();
        println!("parsed!\n");

        println!("checking...");
        self.check();
        println!();
        println!(
            "{}",
            self.cfg
                .as_ref()
                .unwrap()
                .format(self.ctx.as_ref().unwrap())
                .unwrap()
        );
        println!("checked!\n");

        println!("running compiler passes...");
        self.run_passes();
        println!("compiled!\n");

        println!("evaling...");
        let out = if self.typed {
            self.eval_cfg()
        } else {
            self.eval_ast()
        };
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

            eprintln!();
            eprintln!("{} @ program.bappy:{}:{}", message, line_number, start_col);
            eprintln!();
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
            eprintln!();
            eprintln!("{}", message);
            eprintln!();
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

pub fn builtins() -> Vec<Builtin> {
    vec![
        Builtin {
            name: "add",
            args: &["lhs", "rhs"],
            ty: TyName::Func {
                arg_tys: vec![TyName::Int, TyName::Int],
                return_ty: Box::new(TyName::Int),
            },
            layout: FrameLayout {
                frame_size: 0,
                args_size: 24,
                reg_offsets: vec![16, 24],
                reg_sizes: vec![8, 8],
                return_offset: 8,
                alloc_offsets: vec![],
            },
            ast_impl: ast_builtin_add,
            cfg_impl: cfg_builtin_add,
        },
        Builtin {
            name: "sub",
            args: &["lhs", "rhs"],
            ty: TyName::Func {
                arg_tys: vec![TyName::Int, TyName::Int],
                return_ty: Box::new(TyName::Int),
            },
            layout: FrameLayout {
                frame_size: 0,
                args_size: 24,
                reg_offsets: vec![16, 24],
                reg_sizes: vec![8, 8],
                return_offset: 8,
                alloc_offsets: vec![],
            },
            ast_impl: ast_builtin_sub,
            cfg_impl: cfg_builtin_sub,
        },
        Builtin {
            name: "mul",
            args: &["lhs", "rhs"],
            ty: TyName::Func {
                arg_tys: vec![TyName::Int, TyName::Int],
                return_ty: Box::new(TyName::Int),
            },
            layout: FrameLayout {
                frame_size: 0,
                args_size: 24,
                reg_offsets: vec![16, 24],
                reg_sizes: vec![8, 8],
                return_offset: 8,
                alloc_offsets: vec![],
            },
            ast_impl: ast_builtin_mul,
            cfg_impl: cfg_builtin_mul,
        },
        Builtin {
            name: "eq",
            args: &["lhs", "rhs"],
            ty: TyName::Func {
                arg_tys: vec![TyName::Int, TyName::Int],
                return_ty: Box::new(TyName::Bool),
            },
            layout: FrameLayout {
                frame_size: 0,
                args_size: 24,
                reg_offsets: vec![16, 24],
                reg_sizes: vec![8, 8],
                return_offset: 8,
                alloc_offsets: vec![],
            },
            ast_impl: ast_builtin_eq,
            cfg_impl: cfg_builtin_eq,
        },
        Builtin {
            name: "not",
            args: &["rhs"],
            ty: TyName::Func {
                arg_tys: vec![TyName::Bool],
                return_ty: Box::new(TyName::Bool),
            },
            layout: FrameLayout {
                frame_size: 0,
                args_size: 8,
                reg_offsets: vec![8],
                reg_sizes: vec![1],
                return_offset: 7,
                alloc_offsets: vec![],
            },
            ast_impl: ast_builtin_not,
            cfg_impl: cfg_builtin_not,
        },
    ]
}
