use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::{self, Write};

use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{alpha1, alphanumeric1, char, digit1, space0, space1},
    combinator::{map, recognize, rest},
    error::ParseError,
    multi::{many0, separated_list0, separated_list1},
    sequence::pair,
    AsChar, IResult, InputTakeAtPosition, Parser,
};

#[cfg(test)]
mod tests;

//
//
//
//
//
//
//
//
//
// main!
//
//
//
//
//
//
//
//
//
//
//

const MAIN_PROGRAM: &str = r#"
    fn temp () -> fn () -> Int {
        fn inner_temp() -> Int {
            ret 2
        }
        ret inner_temp
    }
    let func: fn() -> fn () -> Int = temp

    if true {
        let capture = 7
        fn outer_capturer() -> fn () -> Int {
            fn inner_capturer() -> Int {
                print capture
                ret capture
            }
            ret inner_capturer
        }
        set func = outer_capturer
    }
    let sub_func = func()
    ret sub_func()
"#;

fn main() {
    Program::typed(MAIN_PROGRAM).run();
}

#[derive(Debug, Clone)]
struct Program<'p> {
    /// Should we use static types?
    typed: bool,

    /// The input we're parsing/checking/executing    
    input: &'p str,
    /// Lines of the input that have been parsed
    input_lines: Vec<(usize, &'p str)>,
    /// Fully parsed `main`
    main: Option<Function<'p>>,

    /// Builtin functions ("the stdlib")
    builtins: Vec<Builtin>,

    /// Printed values resulting from `eval`
    output: Option<String>,
    cur_eval_span: Span,
}

impl<'p> Program<'p> {
    #[allow(dead_code)]
    pub fn untyped(input: &'p str) -> Self {
        let mut out = Self::new(input);
        out.typed = false;
        out
    }

    #[allow(dead_code)]
    pub fn typed(input: &'p str) -> Self {
        let mut out = Self::new(input);
        out.typed = true;
        out
    }

    fn new(input: &'p str) -> Self {
        Self {
            typed: false,
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

    pub fn line_number(&self, span: Span) -> usize {
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

//
//
//
//
//
//
//
//
//
// parser!
//
//
//
//
//
//
//
//
//
//
//

/// A span of the input code.
///
/// Values are absolute addresses, which can be converted
/// into an offset by subtracting the address of the input.
#[derive(Debug, Copy, Clone)]
struct Span {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone)]
struct Function<'p> {
    name: &'p str,
    args: Vec<VarDecl<'p>>,
    stmts: Vec<Statement<'p>>,
    ty: TyName<'p>,
    captures: HashSet<&'p str>,
}

#[derive(Debug, Clone)]
struct StructDecl<'p> {
    name: &'p str,
    fields: Vec<FieldDecl<'p>>,
}

#[derive(Debug, Clone)]
struct Statement<'p> {
    code: Stmt<'p>,
    span: Span,
}

#[derive(Debug, Clone)]
struct VarPath<'p> {
    ident: &'p str,
    fields: Vec<&'p str>,
}

#[derive(Debug, Clone)]
enum Stmt<'p> {
    If {
        expr: Expression<'p>,
        stmts: Vec<Statement<'p>>,
        else_stmts: Vec<Statement<'p>>,
    },
    Loop {
        stmts: Vec<Statement<'p>>,
    },
    Let {
        name: VarDecl<'p>,
        expr: Expression<'p>,
    },
    Set {
        path: VarPath<'p>,
        expr: Expression<'p>,
    },
    Func {
        func: Function<'p>,
    },
    Struct(StructDecl<'p>),
    Ret {
        expr: Expression<'p>,
    },
    Print {
        expr: Expression<'p>,
    },
    Break,
    Continue,
}

#[derive(Debug, Clone)]
struct Expression<'p> {
    code: Expr<'p>,
    span: Span,
}

#[derive(Debug, Clone)]
enum Expr<'p> {
    Call {
        func: &'p str,
        args: Vec<Expression<'p>>,
    },
    Lit(Literal<'p>),
    VarPath(VarPath<'p>),
    Tuple(Vec<Expression<'p>>),
    Named {
        name: &'p str,
        args: Vec<(&'p str, Expression<'p>)>,
    },
}

#[derive(Debug, Clone)]
enum Literal<'p> {
    Int(i64),
    Str(&'p str),
    Bool(bool),
    Empty(()),
}

impl Literal<'_> {
    fn ty(&self) -> TyName<'static> {
        match self {
            Literal::Int(_) => TyName::Int,
            Literal::Str(_) => TyName::Str,
            Literal::Bool(_) => TyName::Bool,
            Literal::Empty(_) => TyName::Empty,
        }
    }
}

#[derive(Debug, Clone)]
struct VarDecl<'p> {
    ident: &'p str,
    ty: TyName<'p>,
    span: Span,
}
#[derive(Debug, Clone)]
struct FieldDecl<'p> {
    ident: &'p str,
    ty: TyName<'p>,
}

// Parse intermediates
enum Item<'p> {
    Comment(&'p str),
    Struct(&'p str),
    Func(&'p str, Vec<VarDecl<'p>>, TyName<'p>),
    Stmt(Stmt<'p>),
    If(Expression<'p>),
    Loop,
    Else,
    End,
}
enum StructItem<'p> {
    Field(FieldDecl<'p>),
    End,
}

struct Block<'p>(Vec<Statement<'p>>);

#[derive(Clone)]
struct Builtin {
    name: &'static str,
    args: &'static [&'static str],
    ty: TyName<'static>,
    func: for<'e, 'p> fn(args: &[Val<'e, 'p>]) -> Val<'e, 'p>,
}

impl fmt::Debug for Builtin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<builtin>")
    }
}

impl<'p> Program<'p> {
    fn parse(&mut self) -> IResult<&'p str, ()> {
        let (i, (Block(stmts), terminal)) = self.parse_block(self.input)?;
        self.main = Some(Function {
            name: "main",
            args: Vec::new(),
            stmts,
            ty: TyName::Func {
                arg_tys: vec![],
                return_ty: Box::new(TyName::Int),
            },
            captures: HashSet::new(),
        });

        if !matches!(terminal, Item::End) {
            self.error(
                format!("Parse Error: `fn` ending eith an `else`"),
                Span {
                    start: addr(i),
                    end: addr(i),
                },
            )
        }

        Ok((i, ()))
    }

    fn parse_block(&mut self, mut i: &'p str) -> IResult<&'p str, (Block<'p>, Item<'p>)> {
        let mut stmts = Vec::new();

        loop {
            if i.trim().is_empty() {
                return Ok((i, (Block(stmts), Item::End)));
            }

            let (new_i, line) = take_until("\n")(i)?;
            self.input_lines.push((addr(line), line));
            println!("{}", line);
            i = &new_i[1..];
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            let stmt_start = addr(line);
            let (rest_of_line, item) = item(line)?;
            let stmt_end = addr(rest_of_line);

            let stmt = match item {
                Item::Struct(name) => {
                    let (new_i, fields) = self.parse_struct_body(i)?;
                    i = new_i;

                    Stmt::Struct(StructDecl { name, fields })
                }
                Item::Func(name, args, return_ty) => {
                    let (new_i, (Block(block_stmts), terminal)) = self.parse_block(i)?;
                    i = new_i;

                    if !matches!(terminal, Item::End) {
                        self.error(
                            format!("Parse Error: `fn` ending with an `else`"),
                            Span {
                                start: addr(i),
                                end: addr(i),
                            },
                        )
                    }

                    Stmt::Func {
                        func: Function {
                            ty: TyName::Func {
                                arg_tys: args.iter().map(|decl| decl.ty.clone()).collect(),
                                return_ty: Box::new(return_ty),
                            },
                            name,
                            args,
                            stmts: block_stmts,
                            // Captures are populated by the type checker
                            captures: HashSet::new(),
                        },
                    }
                }
                Item::If(expr) => {
                    let (new_i, (Block(block_stmts), terminal)) = self.parse_block(i)?;
                    i = new_i;

                    let else_stmts = if let Item::Else = terminal {
                        let (new_i, (Block(else_stmts), terminal)) = self.parse_block(i)?;
                        i = new_i;

                        if !matches!(terminal, Item::End) {
                            self.error(
                                format!("Parse Error: `else` ending with an `else`"),
                                Span {
                                    start: addr(i),
                                    end: addr(i),
                                },
                            )
                        }
                        else_stmts
                    } else {
                        Vec::new()
                    };

                    Stmt::If {
                        expr,
                        stmts: block_stmts,
                        else_stmts,
                    }
                }
                Item::Loop => {
                    let (new_i, (Block(block_stmts), terminal)) = self.parse_block(i)?;
                    i = new_i;

                    if !matches!(terminal, Item::End) {
                        self.error(
                            format!("Parse Error: `loop` ending with an `else`"),
                            Span {
                                start: addr(i),
                                end: addr(i),
                            },
                        )
                    }

                    Stmt::Loop { stmts: block_stmts }
                }
                Item::Stmt(stmt) => stmt,
                Item::Comment(_comment) => {
                    // discard it
                    continue;
                }
                item @ Item::End | item @ Item::Else => return Ok((i, (Block(stmts), item))),
            };

            stmts.push(Statement {
                code: stmt,
                span: Span {
                    start: stmt_start,
                    end: stmt_end,
                },
            });
        }
    }
    fn parse_struct_body(&mut self, mut i: &'p str) -> IResult<&'p str, Vec<FieldDecl<'p>>> {
        let mut fields = Vec::new();
        loop {
            if i.trim().is_empty() {
                return Ok((i, fields));
            }

            let (new_i, line) = take_until("\n")(i)?;
            self.input_lines.push((addr(line), line));
            println!("{}", line);
            i = &new_i[1..];
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            // let stmt_start = addr(line);
            let (_rest_of_line, item) = struct_item(line)?;
            // let stmt_end = addr(rest_of_line);

            match item {
                StructItem::Field(field) => {
                    fields.push(field);
                }
                StructItem::End => return Ok((i, fields)),
            }
        }
    }
}

fn addr(input: &str) -> usize {
    input.as_ptr() as usize
}

fn item(i: &str) -> IResult<&str, Item> {
    alt((
        item_comment,
        item_else,
        item_end,
        item_if,
        item_loop,
        item_func,
        item_struct,
        item_stmt,
    ))(i)
}

fn item_comment(i: &str) -> IResult<&str, Item> {
    let (i, _) = tag("//")(i)?;
    let (i, comment) = rest(i)?;

    Ok((i, Item::Comment(comment)))
}

fn item_func(i: &str) -> IResult<&str, Item> {
    let (i, _) = tag("fn")(i)?;
    let (i, _) = space1(i)?;
    let (i, name) = ident(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag("(")(i)?;
    let (i, _) = space0(i)?;
    let (i, args) = separated_list0(char(','), padded(var_decl))(i)?;
    let (i, _) = tag(")")(i)?;
    let (i, return_ty) = if let Ok((i, return_ty)) = return_ty(i) {
        (i, return_ty)
    } else {
        (i, TyName::Unknown)
    };
    let (i, _) = space0(i)?;
    let (i, _) = tag("{")(i)?;

    Ok((i, Item::Func(name, args, return_ty)))
}

fn item_struct(i: &str) -> IResult<&str, Item> {
    let (i, _) = tag("struct")(i)?;
    let (i, _) = space1(i)?;
    let (i, name) = ident(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag("{")(i)?;

    Ok((i, Item::Struct(name)))
}

fn return_ty(i: &str) -> IResult<&str, TyName> {
    let (i, _) = space0(i)?;
    let (i, _) = tag("->")(i)?;
    let (i, _) = space0(i)?;
    ty_ref(i)
}

fn item_if(i: &str) -> IResult<&str, Item> {
    let (i, _) = tag("if")(i)?;
    let (i, _) = space1(i)?;
    let (i, expr) = expr(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag("{")(i)?;
    Ok((i, Item::If(expr)))
}

fn item_else(i: &str) -> IResult<&str, Item> {
    let (i, _) = tag("}")(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag("else")(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag("{")(i)?;
    Ok((i, Item::Else))
}

fn item_loop(i: &str) -> IResult<&str, Item> {
    let (i, _) = tag("loop")(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag("{")(i)?;
    Ok((i, Item::Loop))
}

fn item_end(i: &str) -> IResult<&str, Item> {
    let (i, _) = tag("}")(i)?;
    Ok((i, Item::End))
}

fn item_stmt(i: &str) -> IResult<&str, Item> {
    map(
        alt((
            stmt_break,
            stmt_continue,
            stmt_let,
            stmt_set,
            stmt_return,
            stmt_print,
        )),
        Item::Stmt,
    )(i)
}

fn stmt_let(i: &str) -> IResult<&str, Stmt> {
    let (i, _) = tag("let")(i)?;
    let (i, _) = space1(i)?;
    let (i, name) = var_decl(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag("=")(i)?;
    let (i, _) = space0(i)?;
    let (i, expr) = expr(i)?;

    Ok((i, Stmt::Let { name, expr }))
}

fn stmt_set(i: &str) -> IResult<&str, Stmt> {
    let (i, _) = tag("set")(i)?;
    let (i, _) = space1(i)?;
    let (i, path) = var_path(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag("=")(i)?;
    let (i, _) = space0(i)?;
    let (i, expr) = expr(i)?;

    Ok((i, Stmt::Set { path, expr }))
}

fn stmt_return(i: &str) -> IResult<&str, Stmt> {
    let (i, _) = tag("ret")(i)?;
    let (i, _) = space1(i)?;
    let (i, expr) = expr(i)?;

    Ok((i, Stmt::Ret { expr }))
}

fn stmt_break(i: &str) -> IResult<&str, Stmt> {
    let (i, _) = tag("break")(i)?;
    Ok((i, Stmt::Break))
}

fn stmt_continue(i: &str) -> IResult<&str, Stmt> {
    let (i, _) = tag("continue")(i)?;
    Ok((i, Stmt::Continue))
}

fn stmt_print(i: &str) -> IResult<&str, Stmt> {
    let (i, _) = tag("print")(i)?;
    let (i, _) = space1(i)?;
    let (i, expr) = expr(i)?;

    Ok((i, Stmt::Print { expr }))
}

fn expr(i: &str) -> IResult<&str, Expression> {
    let start_of_expr = addr(i);
    let (i, expr) = alt((expr_tuple, expr_named, expr_call, expr_lit, expr_var_path))(i)?;
    let end_of_expr = addr(i);

    Ok((
        i,
        Expression {
            code: expr,
            span: Span {
                start: start_of_expr,
                end: end_of_expr,
            },
        },
    ))
}

fn expr_call(i: &str) -> IResult<&str, Expr> {
    let (i, func) = ident(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag("(")(i)?;
    let (i, _) = space0(i)?;
    let (i, args) = separated_list0(char(','), padded(expr))(i)?;
    let (i, _) = tag(")")(i)?;

    Ok((i, Expr::Call { func, args }))
}

fn expr_var_path(i: &str) -> IResult<&str, Expr> {
    map(var_path, Expr::VarPath)(i)
}

fn var_path(i: &str) -> IResult<&str, VarPath> {
    let (i, name) = ident(i)?;
    let (i, _) = space0(i)?;

    let res: IResult<_, _> = padded(tag("."))(i);
    let (i, fields) = if let Ok((i, _)) = res {
        separated_list1(char('.'), padded(alt((ident, digit1))))(i)?
    } else {
        (i, Vec::new())
    };

    Ok((
        i,
        VarPath {
            ident: name,
            fields,
        },
    ))
}

fn expr_lit(i: &str) -> IResult<&str, Expr> {
    map(alt((lit_int, lit_str, lit_bool, lit_empty)), Expr::Lit)(i)
}

fn expr_tuple(i: &str) -> IResult<&str, Expr> {
    let (i, _) = tag("(")(i)?;
    let (i, _) = space0(i)?;
    let (i, args) = separated_list1(char(','), padded(expr))(i)?;
    let (i, _) = tag(")")(i)?;

    Ok((i, Expr::Tuple(args)))
}

fn expr_named(i: &str) -> IResult<&str, Expr> {
    let (i, name) = ident(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag("{")(i)?;
    let (i, _) = space0(i)?;
    let (i, args) = separated_list0(char(','), padded(expr_named_init))(i)?;
    let (i, _) = tag("}")(i)?;

    Ok((i, Expr::Named { name, args }))
}

fn expr_named_init(i: &str) -> IResult<&str, (&str, Expression)> {
    let (i, id) = ident(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag(":")(i)?;
    let (i, _) = space0(i)?;
    let (i, expr) = expr(i)?;

    Ok((i, (id, expr)))
}

fn lit_int(i: &str) -> IResult<&str, Literal> {
    map(nom::character::complete::i64, Literal::Int)(i)
}

fn lit_str(i: &str) -> IResult<&str, Literal> {
    let (i, _) = tag("\"")(i)?;
    let (i, string) = take_until("\"")(i)?;
    let (i, _) = tag("\"")(i)?;
    Ok((i, Literal::Str(string)))
}

fn lit_bool(i: &str) -> IResult<&str, Literal> {
    alt((
        map(tag("true"), |_| Literal::Bool(true)),
        map(tag("false"), |_| Literal::Bool(false)),
    ))(i)
}

fn lit_empty(i: &str) -> IResult<&str, Literal> {
    map(tag("()"), |_| Literal::Empty(()))(i)
}

fn var_decl(i: &str) -> IResult<&str, VarDecl> {
    alt((typed_ident, untyped_ident))(i)
}

fn ident(i: &str) -> IResult<&str, &str> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ))(i)
}

fn untyped_ident(i: &str) -> IResult<&str, VarDecl> {
    let start = addr(i);
    let (i, id) = ident(i)?;
    let end = addr(i);
    let span = Span { start, end };
    Ok((
        i,
        VarDecl {
            ident: id,
            ty: TyName::Unknown,
            span,
        },
    ))
}

fn typed_ident(i: &str) -> IResult<&str, VarDecl> {
    let start = addr(i);
    let (i, id) = ident(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag(":")(i)?;
    let (i, _) = space0(i)?;
    let (i, ty) = ty_ref(i)?;
    let end = addr(i);
    let span = Span { start, end };

    Ok((
        i,
        VarDecl {
            ident: id,
            ty,
            span,
        },
    ))
}

fn ty_ref(i: &str) -> IResult<&str, TyName> {
    alt((
        ty_ref_int,
        ty_ref_str,
        ty_ref_empty,
        ty_ref_bool,
        ty_ref_func,
        ty_ref_tuple,
        ty_ref_named,
    ))(i)
}

fn ty_ref_int(i: &str) -> IResult<&str, TyName> {
    map(tag("Int"), |_| TyName::Int)(i)
}
fn ty_ref_str(i: &str) -> IResult<&str, TyName> {
    map(tag("Str"), |_| TyName::Str)(i)
}
fn ty_ref_bool(i: &str) -> IResult<&str, TyName> {
    map(tag("Bool"), |_| TyName::Bool)(i)
}
fn ty_ref_empty(i: &str) -> IResult<&str, TyName> {
    map(tag("()"), |_| TyName::Empty)(i)
}
fn ty_ref_func(i: &str) -> IResult<&str, TyName> {
    let (i, _) = tag("fn")(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag("(")(i)?;
    let (i, _) = space0(i)?;
    let (i, arg_tys) = separated_list0(char(','), padded(ty_ref))(i)?;
    let (i, _) = tag(")")(i)?;
    let (i, return_ty) = return_ty(i)?;

    Ok((
        i,
        TyName::Func {
            arg_tys,
            return_ty: Box::new(return_ty),
        },
    ))
}
fn ty_ref_tuple(i: &str) -> IResult<&str, TyName> {
    let (i, _) = tag("(")(i)?;
    let (i, _) = space0(i)?;
    let (i, tys) = separated_list1(char(','), padded(ty_ref))(i)?;
    let (i, _) = tag(")")(i)?;

    Ok((i, TyName::Tuple(tys)))
}
fn ty_ref_named(i: &str) -> IResult<&str, TyName> {
    map(ident, |name| TyName::Named(name))(i)
}

fn struct_item(i: &str) -> IResult<&str, StructItem> {
    alt((struct_item_field, struct_item_end))(i)
}
fn struct_item_field(i: &str) -> IResult<&str, StructItem> {
    let (i, id) = ident(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag(":")(i)?;
    let (i, _) = space0(i)?;
    let (i, ty) = ty_ref(i)?;

    Ok((i, StructItem::Field(FieldDecl { ident: id, ty })))
}
fn struct_item_end(i: &str) -> IResult<&str, StructItem> {
    let (i, _) = tag("}")(i)?;
    // TODO: require rest of line to be whitespace
    Ok((i, StructItem::End))
}

pub fn padded<F, T, O, E>(mut parser: F) -> impl FnMut(T) -> IResult<T, O, E>
where
    F: Parser<T, O, E>,
    T: InputTakeAtPosition,
    E: ParseError<T>,
    <T as InputTakeAtPosition>::Item: AsChar + Clone,
{
    move |i| {
        let (i, _) = space0(i)?;
        let (i, val) = parser.parse(i)?;
        let (i, _) = space0(i)?;

        Ok((i, val))
    }
}

//
//
//
//
//
//
//
//
//
// static analysis!
//
//
//
//
//
//
//
//
//
//
//

/// "Names" of types -- the raw output of the parser.
///
/// Names should not be used directly for type resolution,
/// because Nominal types (TyName::Named) will mess things up.
///
/// i.e. you don't know which `Point` the type `(Point, Point)` refers to.
/// With things like inference, they might be different `Point` types!
///
/// Instead you should use the TyCtx to convert a TyName to
/// a `Ty` and `TyIdx` which handle nominal types properly.
#[derive(Clone, Debug)]
enum TyName<'p> {
    Int,
    Str,
    Bool,
    Empty,
    Func {
        arg_tys: Vec<TyName<'p>>,
        return_ty: Box<TyName<'p>>,
    },
    Tuple(Vec<TyName<'p>>),
    Named(&'p str),
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Ty<'p> {
    Int,
    Str,
    Bool,
    Empty,
    Func {
        arg_tys: Vec<TyIdx>,
        return_ty: TyIdx,
    },
    Tuple(Vec<TyIdx>),
    NamedStruct(StructTy<'p>),
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StructTy<'p> {
    name: &'p str,
    fields: Vec<FieldTy<'p>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FieldTy<'p> {
    ident: &'p str,
    ty: TyIdx,
}

type TyIdx = usize;

/// Information on all the types.
struct TyCtx<'p> {
    /// Whether static types are enabled/enforced.
    is_typed: bool,
    /// The list of every known type.
    ///
    /// These are the "canonical" copies of each type. Types are
    /// registered here via `memoize`, which returns a TyIdx into
    /// this array.
    ///
    /// Types should be compared by checking if they have the same
    /// TyIdx. This allows you to properly compare nominal types
    /// in the face of shadowing and similar situations.
    tys: Vec<Ty<'p>>,

    /// Mappings from structural types we've seen to type indices.
    ///
    /// This is used to get the canonical TyIdx of a structural type
    /// (including builtin primitives).
    ///
    /// Nominal types (structs) are stored in `envs`, because they
    /// go in and out of scope.
    ty_map: HashMap<Ty<'p>, TyIdx>,

    /// Scoped type info, reflecting the fact that struct definitions
    /// and variables come in and out of scope.
    ///
    /// These values are "cumulative", so type names and variables
    /// should be looked up by searching backwards in this array.
    ///
    /// If nothing is found, that type name / variable name is undefined
    /// at this point in the program.
    envs: Vec<CheckEnv<'p>>,

    /// The absence of a type annotation, saved for easy comparison.
    ty_unknown: TyIdx,
    /// The empty tuple, saved for easy use.
    ty_empty: TyIdx,
}

/// Information about types for a specific scope.
struct CheckEnv<'p> {
    /// The types of variables
    vars: HashMap<&'p str, TyIdx>,
    /// The struct definitions and TyIdx's
    tys: HashMap<&'p str, TyIdx>,

    /// Whether this scope is the root of a function
    /// (the scope of its arguments). If you walk over
    /// this kind of frame, then you're accessing captures.
    is_function_root: bool,
}
struct CheckEntry<'a, 'p> {
    capture_depth: usize,
    entry: std::collections::hash_map::OccupiedEntry<'a, &'p str, TyIdx>,
}

impl<'p> TyCtx<'p> {
    fn resolve_var<'a>(&'a mut self, var_name: &'p str) -> Option<CheckEntry<'a, 'p>> {
        // By default we're accessing locals
        let mut capture_depth = 0;
        use std::collections::hash_map::Entry;
        for env in self.envs.iter_mut().rev() {
            if let Entry::Occupied(entry) = env.vars.entry(var_name) {
                return Some(CheckEntry {
                    capture_depth,
                    entry,
                });
            }
            if env.is_function_root {
                // We're walking over a function root, so we're now
                // accessing captures. We track this with an integer
                // because if we capture multiple functions deep, our
                // ancestor functions also need to capture that value.
                capture_depth += 1;
            }
        }
        None
    }

    fn push_struct_decl(
        &mut self,
        program: &mut Program<'p>,
        struct_decl: StructDecl<'p>,
    ) -> TyIdx {
        let fields = struct_decl
            .fields
            .iter()
            .map(|f| FieldTy {
                ident: f.ident,
                ty: self.memoize_ty(program, &f.ty),
            })
            .collect();
        let ty_idx = self.tys.len();
        self.tys.push(Ty::NamedStruct(StructTy {
            name: struct_decl.name,
            fields,
        }));
        self.envs
            .last_mut()
            .unwrap()
            .tys
            .insert(struct_decl.name, ty_idx);
        ty_idx
    }

    fn resolve_nominal_ty<'a>(&'a mut self, ty_name: &'p str) -> Option<TyIdx> {
        if self.is_typed {
            for (_depth, env) in self.envs.iter_mut().rev().enumerate() {
                if let Some(ty) = env.tys.get(ty_name) {
                    return Some(*ty);
                }
            }
            None
        } else {
            None
        }
    }

    fn memoize_ty(&mut self, program: &mut Program<'p>, ty_name: &TyName<'p>) -> TyIdx {
        if self.is_typed {
            match ty_name {
                TyName::Int => self.memoize_inner(Ty::Int),
                TyName::Str => self.memoize_inner(Ty::Str),
                TyName::Bool => self.memoize_inner(Ty::Bool),
                TyName::Empty => self.memoize_inner(Ty::Empty),
                TyName::Unknown => self.memoize_inner(Ty::Unknown),
                TyName::Func { arg_tys, return_ty } => {
                    let arg_tys = arg_tys
                        .iter()
                        .map(|arg_ty_name| self.memoize_ty(program, arg_ty_name))
                        .collect();
                    let mut return_ty = self.memoize_ty(program, return_ty);

                    // Allow the return type to be elided
                    if return_ty == self.ty_unknown {
                        return_ty = self.ty_empty;
                    }
                    self.memoize_inner(Ty::Func { arg_tys, return_ty })
                }
                TyName::Tuple(arg_ty_names) => {
                    let arg_tys = arg_ty_names
                        .iter()
                        .map(|arg_ty_name| self.memoize_ty(program, arg_ty_name))
                        .collect();
                    self.memoize_inner(Ty::Tuple(arg_tys))
                }
                TyName::Named(name) => {
                    // Nominal types take a separate path because they're scoped
                    if let Some(ty_idx) = self.resolve_nominal_ty(name) {
                        ty_idx
                    } else {
                        // FIXME: rejig this so the line info is better
                        program.error(
                            format!("Compile Error: use of undefined type name: {}", name),
                            Span {
                                start: addr(program.input),
                                end: addr(program.input),
                            },
                        )
                    }
                }
            }
        } else {
            0
        }
    }

    fn memoize_inner(&mut self, ty: Ty<'p>) -> TyIdx {
        if let Some(idx) = self.ty_map.get(&ty) {
            *idx
        } else {
            let ty1 = ty.clone();
            let ty2 = ty;
            let idx = self.tys.len();
            self.ty_map.insert(ty1, idx);
            self.tys.push(ty2);
            idx
        }
    }

    fn realize_ty(&self, ty: TyIdx) -> &Ty<'p> {
        if self.is_typed {
            self.tys
                .get(ty)
                .expect("Internal Compiler Error: invalid TyIdx")
        } else {
            &Ty::Unknown
        }
    }

    fn format_ty(&self, ty: TyIdx) -> String {
        match self.realize_ty(ty) {
            Ty::Int => format!("Int"),
            Ty::Str => format!("Str"),
            Ty::Bool => format!("Bool"),
            Ty::Empty => format!("()"),
            Ty::Unknown => format!("<unknown>"),
            Ty::NamedStruct(struct_decl) => format!("{}", struct_decl.name),
            Ty::Tuple(arg_tys) => {
                let mut f = String::new();
                write!(f, "(").unwrap();
                for (idx, arg_ty_idx) in arg_tys.iter().enumerate() {
                    if idx != 0 {
                        write!(f, ", ").unwrap();
                    }
                    let arg = self.format_ty(*arg_ty_idx);
                    write!(f, "{}", arg).unwrap();
                }
                write!(f, ")").unwrap();
                f
            }
            Ty::Func { arg_tys, return_ty } => {
                let mut f = String::new();
                write!(f, "fn (").unwrap();
                for (idx, arg_ty_idx) in arg_tys.iter().enumerate() {
                    if idx != 0 {
                        write!(f, ", ").unwrap();
                    }
                    let arg = self.format_ty(*arg_ty_idx);
                    write!(f, "{}", arg).unwrap();
                }
                write!(f, ") -> ").unwrap();

                let ret = self.format_ty(*return_ty);
                write!(f, "{}", ret).unwrap();
                f
            }
        }
    }
}

impl<'p> Program<'p> {
    fn check(&mut self) {
        let mut ctx = TyCtx {
            tys: Vec::new(),
            ty_map: HashMap::new(),
            envs: Vec::new(),
            is_typed: self.typed,
            ty_unknown: 0,
            ty_empty: 0,
        };

        // Cache some key types
        ctx.ty_unknown = ctx.memoize_inner(Ty::Unknown);
        ctx.ty_empty = ctx.memoize_inner(Ty::Empty);

        // Set up globals (stdlib)
        let builtins = self
            .builtins
            .clone()
            .iter()
            .map(|builtin| (builtin.name, ctx.memoize_ty(self, &builtin.ty)))
            .collect();
        let globals = CheckEnv {
            vars: builtins,
            tys: HashMap::new(),
            // Doesn't really matter what this value is for the globals
            is_function_root: false,
        };
        ctx.envs.push(globals);
        let mut captures = Vec::new();

        let mut main = self.main.take().unwrap();
        self.check_func(&mut main, &mut ctx, &mut captures);

        if ctx.envs.len() != 1 {
            self.error(
                format!("Internal Compiler Error: scopes were improperly popped"),
                Span {
                    start: addr(self.input),
                    end: addr(self.input),
                },
            );
        }

        self.main = Some(main);
    }

    fn check_func(
        &mut self,
        func: &mut Function<'p>,
        ctx: &mut TyCtx<'p>,
        captures: &mut Vec<HashSet<&'p str>>,
    ) {
        let vars = func
            .args
            .iter()
            .map(|decl| {
                let ty = ctx.memoize_ty(self, &decl.ty);
                if !self.typed || ty != ctx.ty_unknown {
                    (decl.ident, ty)
                } else {
                    self.error(
                        format!("Compile Error: function arguments must have types"),
                        decl.span,
                    )
                }
            })
            .collect();

        ctx.envs.push(CheckEnv {
            vars,
            tys: HashMap::new(),
            is_function_root: true,
        });

        captures.push(HashSet::new());

        let return_ty = if let TyName::Func { return_ty, .. } = &func.ty {
            return_ty
        } else {
            panic!(
                "Internal Compiler Error: function that wasn't a function? {}",
                func.name
            );
        };

        // If the `-> Type` is omitted from a function decl, assume
        // the empty tuple `()`, just like Rust.
        let mut return_ty = ctx.memoize_ty(self, return_ty);
        if return_ty == ctx.ty_unknown {
            return_ty = ctx.ty_empty;
        }

        self.check_block(&mut func.stmts, ctx, captures, return_ty);

        func.captures = captures.pop().unwrap();
        ctx.envs.pop();
    }

    fn check_block(
        &mut self,
        stmts: &mut [Statement<'p>],
        ctx: &mut TyCtx<'p>,
        captures: &mut Vec<HashSet<&'p str>>,
        return_ty: TyIdx,
    ) {
        ctx.envs.push(CheckEnv {
            vars: HashMap::new(),
            tys: HashMap::new(),
            is_function_root: false,
        });
        for Statement {
            code: stmt,
            span: stmt_span,
        } in stmts
        {
            match stmt {
                Stmt::If {
                    expr,
                    stmts,
                    else_stmts,
                } => {
                    let expr_ty = self.check_expr(expr, ctx, captures);
                    let expected_ty = ctx.memoize_ty(self, &TyName::Bool);

                    self.check_ty(ctx, expr_ty, expected_ty, "`if`", expr.span);
                    self.check_block(stmts, ctx, captures, return_ty);
                    self.check_block(else_stmts, ctx, captures, return_ty);
                }
                Stmt::Loop { stmts } => {
                    self.check_block(stmts, ctx, captures, return_ty);
                }
                Stmt::Struct(struct_decl) => {
                    ctx.push_struct_decl(self, struct_decl.clone());
                }
                Stmt::Func { func } => {
                    // We push a func's name after checking it to avoid
                    // infinite capture recursion. This means naive recursion
                    // is illegal.
                    self.check_func(func, ctx, captures);
                    let func_ty = ctx.memoize_ty(self, &func.ty);
                    ctx.envs.last_mut().unwrap().vars.insert(func.name, func_ty);
                }
                Stmt::Let { name, expr } => {
                    let expr_ty = self.check_expr(expr, ctx, captures);
                    let expected_ty = ctx.memoize_ty(self, &name.ty);

                    // If a let statement has no type annotation, infer it
                    // to have the type of the expr assigned to it. Ultimately
                    // this just means not bothering to type check it.
                    if expected_ty != ctx.ty_unknown {
                        self.check_ty(ctx, expr_ty, expected_ty, "`let`", expr.span);
                    }

                    ctx.envs
                        .last_mut()
                        .unwrap()
                        .vars
                        .insert(name.ident, expr_ty);
                }
                Stmt::Set {
                    path: var_path,
                    expr,
                } => {
                    let expr_ty = self.check_expr(expr, ctx, captures);
                    if let Some(var) = ctx.resolve_var(var_path.ident) {
                        if var.capture_depth == 0 {
                            let var_ty = *var.entry.get();
                            let expected_ty =
                                self.resolve_var_path(ctx, var_ty, &var_path.fields, *stmt_span);
                            self.check_ty(ctx, expr_ty, expected_ty, "`set`", expr.span);
                        } else {
                            self.error(
                                format!("Compile Error: Trying to `set` captured variable '{}' (captures are by-value!)", var_path.ident),
                                *stmt_span,
                            )
                        }
                    } else {
                        self.error(
                            format!(
                                "Compile Error: Trying to `set` undefined variable '{}'",
                                var_path.ident,
                            ),
                            *stmt_span,
                        )
                    }
                }
                Stmt::Ret { expr } => {
                    let expr_ty = self.check_expr(expr, ctx, captures);

                    self.check_ty(ctx, expr_ty, return_ty, "return", expr.span);
                }
                Stmt::Print { expr } => {
                    let _expr_ty = self.check_expr(expr, ctx, captures);
                    // Print takes any value, it's magic!
                }
                Stmt::Break | Stmt::Continue => {
                    // Nothing to analyze
                }
            }
        }
        ctx.envs.pop();
    }

    fn check_expr(
        &mut self,
        expr: &Expression<'p>,
        ctx: &mut TyCtx<'p>,
        captures: &mut Vec<HashSet<&'p str>>,
    ) -> TyIdx {
        match &expr.code {
            Expr::Lit(lit) => {
                return ctx.memoize_ty(self, &lit.ty());
            }
            Expr::VarPath(var_path) => {
                if let Some(var) = ctx.resolve_var(var_path.ident) {
                    for (captures, _) in captures.iter_mut().rev().zip(0..var.capture_depth) {
                        captures.insert(var_path.ident);
                    }
                    let var_ty = *var.entry.get();
                    self.resolve_var_path(ctx, var_ty, &var_path.fields, expr.span)
                } else {
                    self.error(
                        format!(
                            "Compile Error: Use of undefined variable '{}'",
                            var_path.ident
                        ),
                        expr.span,
                    )
                }
            }
            Expr::Tuple(args) => {
                let arg_tys = args
                    .iter()
                    .map(|arg| self.check_expr(arg, ctx, captures))
                    .collect();
                ctx.memoize_inner(Ty::Tuple(arg_tys))
            }
            Expr::Named { name, args } => {
                if let Some(ty_idx) = ctx.resolve_nominal_ty(name) {
                    if let Ty::NamedStruct(ty_decl) = ctx.realize_ty(ty_idx) {
                        let ty_decl = ty_decl.clone();
                        if args.len() != ty_decl.fields.len() {
                            self.error(
                                format!(
                                    "Compile Error: field count mismatch (expected {}, got {})",
                                    ty_decl.fields.len(),
                                    args.len()
                                ),
                                expr.span,
                            )
                        }
                        for ((field, arg), field_decl) in args.iter().zip(ty_decl.fields.iter()) {
                            if *field != field_decl.ident {
                                self.error(
                                    format!(
                                        "Compile Error: field name mismatch (expected {}, got {})",
                                        field_decl.ident, field
                                    ),
                                    arg.span,
                                )
                            }

                            let expr_ty = self.check_expr(arg, ctx, captures);
                            let expected_ty = field_decl.ty;
                            self.check_ty(ctx, expr_ty, expected_ty, "struct literal", arg.span);
                        }
                        ty_idx
                    } else {
                        let ty_str = ctx.format_ty(ty_idx);
                        panic!(
                            "Internal Compiler Error: nominal ty wasn't nominal?? {}: {}",
                            name, ty_str
                        );
                    }
                } else if self.typed {
                    self.error(
                        format!("Compile Error: Use of undefined struct '{}'", name),
                        expr.span,
                    )
                } else {
                    // In untyped mode it's ok for a struct to be used without being declared!
                    for (_field, arg) in args {
                        self.check_expr(arg, ctx, captures);
                    }
                    return ctx.memoize_ty(self, &TyName::Named(name));
                }
            }
            Expr::Call { func, args } => {
                if let Some(var) = ctx.resolve_var(func) {
                    for (captures, _) in captures.iter_mut().rev().zip(0..var.capture_depth) {
                        captures.insert(func);
                    }

                    let var_ty = *var.entry.get();
                    let func_ty = ctx.realize_ty(var_ty).clone();
                    let (arg_tys, return_ty) = if let Ty::Func { arg_tys, return_ty } = func_ty {
                        (arg_tys.clone(), return_ty)
                    } else if self.typed {
                        self.error(
                            format!("Compile Error: Function call must have Func type!"),
                            expr.span,
                        )
                    } else {
                        (Vec::new(), ctx.memoize_ty(self, &TyName::Unknown))
                    };

                    if self.typed && arg_tys.len() != args.len() {
                        self.error(
                            format!(
                                "Compile Error: arg count mismatch (expected {:?}, got {:?})",
                                arg_tys.len(),
                                args.len(),
                            ),
                            expr.span,
                        )
                    }

                    for (idx, arg) in args.iter().enumerate() {
                        let expr_ty = self.check_expr(arg, ctx, captures);
                        let expected_ty = arg_tys
                            .get(idx)
                            .copied()
                            .unwrap_or(ctx.memoize_ty(self, &TyName::Unknown));

                        self.check_ty(ctx, expr_ty, expected_ty, "arg", arg.span);
                    }
                    return return_ty;
                } else {
                    self.error(
                        format!("Compile Error: Call of undefined function '{}'", func),
                        expr.span,
                    )
                }
            }
        }
    }

    #[track_caller]
    fn check_ty(
        &mut self,
        ctx: &TyCtx<'p>,
        computed_ty: TyIdx,
        expected_ty: TyIdx,
        env_name: &str,
        span: Span,
    ) {
        if self.typed && computed_ty != expected_ty {
            let expected = ctx.format_ty(expected_ty);
            let computed = ctx.format_ty(computed_ty);

            let msg = if expected != computed {
                format!(
                    "Compile Error: {} type mismatch (expected {}, got {})",
                    env_name, expected, computed,
                )
            } else {
                format!(
                    r#"Compile Error: {} type mismatch (expected {}, got {})
NOTE: the types look the same, but the named types have different decls!"#,
                    env_name, expected, computed,
                )
            };
            self.error(msg, span)
        }
    }

    fn resolve_var_path(
        &mut self,
        ctx: &mut TyCtx<'p>,
        root_ty: TyIdx,
        path: &[&'p str],
        span: Span,
    ) -> TyIdx {
        let mut cur_ty = root_ty;
        'path: for field in path {
            match ctx.realize_ty(cur_ty) {
                Ty::NamedStruct(struct_decl) => {
                    for struct_field in &struct_decl.fields {
                        if &struct_field.ident == field {
                            cur_ty = struct_field.ty;
                            continue 'path;
                        }
                    }
                    self.error(
                        format!(
                            "Compile Error: {} is not a field of {}",
                            field, struct_decl.name
                        ),
                        span,
                    )
                }
                Ty::Tuple(arg_tys) => {
                    if let Some(field_ty) =
                        field.parse::<usize>().ok().and_then(|idx| arg_tys.get(idx))
                    {
                        cur_ty = *field_ty;
                        continue 'path;
                    } else {
                        self.error(
                            format!(
                                "Compile Error: {} is not a field of {}",
                                field,
                                ctx.format_ty(cur_ty)
                            ),
                            span,
                        )
                    }
                }
                _ => self.error(
                    format!(
                        "Compile Error: there are no fields on type {}",
                        ctx.format_ty(cur_ty)
                    ),
                    span,
                ),
            }
        }
        cur_ty
    }
}

//
//
//
//
//
//
//
//
//
// builtins!
//
//
//
//
//
//
//
//
//
//
//

fn builtin_add<'e, 'p>(args: &[Val<'e, 'p>]) -> Val<'e, 'p> {
    assert!(
        args.len() == 2,
        "Runtime Error: Builtin add had wrong number of args"
    );
    if let (Val::Int(lhs), Val::Int(rhs)) = (&args[0], &args[1]) {
        Val::Int(lhs + rhs)
    } else {
        panic!("Runtime Error: Builtin add had wrong type of args")
    }
}

fn builtin_mul<'e, 'p>(args: &[Val<'e, 'p>]) -> Val<'e, 'p> {
    assert!(
        args.len() == 2,
        "Runtime Error: Builtin mul had wrong number of args"
    );
    if let (Val::Int(lhs), Val::Int(rhs)) = (&args[0], &args[1]) {
        Val::Int(lhs * rhs)
    } else {
        panic!("Runtime Error: Builtin mul had wrong type of args")
    }
}

fn builtin_sub<'e, 'p>(args: &[Val<'e, 'p>]) -> Val<'e, 'p> {
    assert!(
        args.len() == 2,
        "Runtime Error: Builtin sub had wrong number of args"
    );
    if let (Val::Int(lhs), Val::Int(rhs)) = (&args[0], &args[1]) {
        Val::Int(lhs - rhs)
    } else {
        panic!("Runtime Error: Builtin sub had wrong type of args")
    }
}

fn builtin_eq<'e, 'p>(args: &[Val<'e, 'p>]) -> Val<'e, 'p> {
    assert!(
        args.len() == 2,
        "Runtime Error: Builtin eq had wrong number of args"
    );

    // Currently restricting this to just ints for it to be statically
    // typed, might change my mind later!
    match (&args[0], &args[1]) {
        (Val::Int(lhs), Val::Int(rhs)) => Val::Bool(lhs == rhs),
        _ => {
            panic!("Runtime Error: Builtin eq had wrong type of args")
        }
    }
}

fn builtin_not<'e, 'p>(args: &[Val<'e, 'p>]) -> Val<'e, 'p> {
    assert!(
        args.len() == 1,
        "Runtime Error: Builtin not had wrong number of args"
    );
    if let Val::Bool(rhs) = &args[0] {
        Val::Bool(!rhs)
    } else {
        panic!("Runtime Error: Builtin sub had wrong type of args")
    }
}

fn builtins() -> Vec<Builtin> {
    vec![
        Builtin {
            name: "add",
            args: &["lhs", "rhs"],
            ty: TyName::Func {
                arg_tys: vec![TyName::Int, TyName::Int],
                return_ty: Box::new(TyName::Int),
            },
            func: builtin_add,
        },
        Builtin {
            name: "sub",
            args: &["lhs", "rhs"],
            ty: TyName::Func {
                arg_tys: vec![TyName::Int, TyName::Int],
                return_ty: Box::new(TyName::Int),
            },
            func: builtin_sub,
        },
        Builtin {
            name: "mul",
            args: &["lhs", "rhs"],
            ty: TyName::Func {
                arg_tys: vec![TyName::Int, TyName::Int],
                return_ty: Box::new(TyName::Int),
            },
            func: builtin_mul,
        },
        Builtin {
            name: "eq",
            args: &["lhs", "rhs"],
            ty: TyName::Func {
                arg_tys: vec![TyName::Int, TyName::Int],
                return_ty: Box::new(TyName::Bool),
            },
            func: builtin_eq,
        },
        Builtin {
            name: "not",
            args: &["rhs"],
            ty: TyName::Func {
                arg_tys: vec![TyName::Bool],
                return_ty: Box::new(TyName::Bool),
            },
            func: builtin_not,
        },
    ]
}

//
//
//
//
//
//
//
//
//
// runtime!
//
//
//
//
//
//
//
//
//
//
//

#[derive(Debug, Clone)]
enum Val<'e, 'p> {
    Int(i64),
    Str(&'p str),
    Bool(bool),
    Empty(()),
    Tuple(Vec<Val<'e, 'p>>),
    Struct(&'p str, BTreeMap<&'p str, Val<'e, 'p>>),
    Func(Closure<'e, 'p>),
    Builtin(Builtin),
}

/*
impl Val<'_, '_> {
    fn ty(&self) -> &TyName {
        match self {
            Val::Int(_) => &TyName::Int,
            Val::Str(_) => &TyName::Str,
            Val::Bool(_) => &TyName::Bool,
            Val::Empty(_) => &TyName::Empty,
            Val::Func(closure) => &closure.func.ty,
            Val::Builtin(builtin) => &builtin.ty,
        }
    }
}
*/

#[derive(Debug, Clone)]
struct Closure<'e, 'p> {
    func: &'e Function<'p>,
    captures: HashMap<&'p str, Val<'e, 'p>>,
}

#[derive(Debug, Clone)]
struct Env<'e, 'p> {
    vals: HashMap<&'p str, Val<'e, 'p>>,
}

enum ControlFlow<'e, 'p> {
    Return(Val<'e, 'p>),
    Break,
    Continue,
    None,
}

impl<'p> Program<'p> {
    fn eval(&mut self) -> i64 {
        let builtins = self
            .builtins
            .iter()
            .map(|builtin| (builtin.name, Val::Builtin(builtin.clone())))
            .collect();

        let main = self.main.take().unwrap();
        let mut envs = vec![Env { vals: builtins }];
        let out = self.eval_func(&main, Vec::new(), HashMap::new(), &mut envs);

        if envs.len() != 1 {
            self.error(
                format!(
                    "Runtime Error: not all scopes were properly popped at end of execution! (extra scopes: {})",
                    envs.len() - 1,
                ),
                self.cur_eval_span,
            )
        }

        if let Val::Int(int) = out {
            self.main = Some(main);
            int
        } else {
            self.error(
                format!("Runtime Error: main must evaluate to an int!"),
                self.cur_eval_span,
            )
        }
    }

    fn eval_func<'e>(
        &mut self,
        func: &'e Function<'p>,
        args: Vec<Val<'e, 'p>>,
        captures: HashMap<&'p str, Val<'e, 'p>>,
        envs: &mut Vec<Env<'e, 'p>>,
    ) -> Val<'e, 'p> {
        if func.args.len() != args.len() {
            self.error(
                format!(
                    "Runtime Error: mismatched argument count for fn {} (expected {}, got {})",
                    func.name,
                    func.args.len(),
                    args.len(),
                ),
                self.cur_eval_span,
            )
        }

        let mut vals = func
            .args
            .iter()
            .map(|arg| arg.ident)
            .zip(args.into_iter())
            .collect::<HashMap<_, _>>();

        if vals.len() != func.args.len() {
            self.error(
                format!("Runtime Error: duplicate arg names for fn {}", func.name,),
                self.cur_eval_span,
            )
        }

        vals.extend(captures.into_iter());

        envs.push(Env { vals });
        let result = self.eval_block(&func.stmts, envs);
        envs.pop();

        match result {
            ControlFlow::Return(val) => val,
            ControlFlow::Break => self.error(
                format!("Runtime Error: break used outside of a loop"),
                self.cur_eval_span,
            ),
            ControlFlow::Continue => self.error(
                format!("Runtime Error: continue used outside of a loop"),
                self.cur_eval_span,
            ),
            ControlFlow::None => self.error(
                format!(
                    "Runtime Error: function didn't return a value: {}",
                    func.name,
                ),
                self.cur_eval_span,
            ),
        }
    }

    fn eval_block<'e>(
        &mut self,
        stmts: &'e [Statement<'p>],
        envs: &mut Vec<Env<'e, 'p>>,
    ) -> ControlFlow<'e, 'p> {
        envs.push(Env {
            vals: HashMap::new(),
        });
        for Statement {
            code: stmt,
            span: stmt_span,
        } in stmts
        {
            self.cur_eval_span = *stmt_span;
            match stmt {
                Stmt::Let { name, expr } => {
                    let val = self.eval_expr(expr, envs);
                    envs.last_mut().unwrap().vals.insert(name.ident, val);
                }
                Stmt::Set {
                    path: var_path,
                    expr,
                } => {
                    let val = self.eval_expr(expr, envs);

                    let base_val = self.eval_resolve_var(var_path.ident, envs);
                    let sub_val = self.eval_resolve_var_path(base_val, &var_path.fields, expr.span);
                    *sub_val = val;
                }
                Stmt::Struct(_struct_decl) => {
                    // TODO: ?
                }
                Stmt::Func { func } => {
                    let captures = func
                        .captures
                        .iter()
                        .map(|&var| (var, self.eval_resolve_var(var, envs).clone()))
                        .collect();

                    envs.last_mut()
                        .unwrap()
                        .vals
                        .insert(func.name, Val::Func(Closure { captures, func }));
                }
                Stmt::If {
                    expr,
                    stmts,
                    else_stmts,
                } => {
                    let result = match self.eval_expr(expr, envs) {
                        Val::Bool(true) => self.eval_block(stmts, envs),
                        Val::Bool(false) => self.eval_block(else_stmts, envs),
                        val => {
                            let val = self.format_val(&val, true, 0);
                            self.error(
                                format!("Runtime Error: Tried to branch on non-boolean {}", val),
                                expr.span,
                            )
                        }
                    };

                    match result {
                        ControlFlow::None => { /* do nothing */ }
                        // All other control flow ends the block immediately
                        flow => {
                            envs.pop();
                            return flow;
                        }
                    }
                }
                Stmt::Loop { stmts } => {
                    loop {
                        let result = self.eval_block(stmts, envs);
                        match result {
                            ControlFlow::Return(val) => {
                                envs.pop();
                                return ControlFlow::Return(val);
                            }
                            ControlFlow::Break => break,
                            ControlFlow::Continue => continue,
                            ControlFlow::None => { /* do nothing */ }
                        }
                    }
                }
                Stmt::Print { expr } => {
                    let val = self.eval_expr(expr, envs);
                    self.print_val(&val);
                }
                Stmt::Ret { expr } => {
                    let val = self.eval_expr(expr, envs);
                    envs.pop();
                    return ControlFlow::Return(val);
                }
                Stmt::Break => {
                    envs.pop();
                    return ControlFlow::Break;
                }
                Stmt::Continue => {
                    envs.pop();
                    return ControlFlow::Continue;
                }
            }
        }

        envs.pop();
        // Nothing special happened, continue execution
        ControlFlow::None
    }

    fn eval_expr<'e>(&mut self, expr: &Expression<'p>, envs: &mut Vec<Env<'e, 'p>>) -> Val<'e, 'p> {
        self.cur_eval_span = expr.span;
        match &expr.code {
            Expr::Call {
                func: func_name,
                args,
            } => {
                let func = self.eval_resolve_var(func_name, envs).clone();
                let evaled_args = args.iter().map(|expr| self.eval_expr(expr, envs)).collect();

                match func {
                    Val::Func(closure) => {
                        self.eval_func(closure.func, evaled_args, closure.captures, envs)
                    }
                    Val::Builtin(builtin) => (builtin.func)(&evaled_args),
                    _ => {
                        let val = self.format_val(&func, true, 0);
                        self.error(
                            format!(
                                "Runtime Error: Tried to call a non-function {}: {}",
                                func_name, val,
                            ),
                            expr.span,
                        )
                    }
                }
            }
            Expr::Tuple(args) => {
                let evaled_args = args.iter().map(|arg| self.eval_expr(arg, envs)).collect();
                Val::Tuple(evaled_args)
            }
            Expr::Named { name, args } => {
                let evaled_fields = args
                    .iter()
                    .map(|(field, expr)| (*field, self.eval_expr(expr, envs)))
                    .collect();
                Val::Struct(name, evaled_fields)
            }
            Expr::VarPath(var_path) => {
                let base_val = self.eval_resolve_var(var_path.ident, envs);
                let sub_val = self.eval_resolve_var_path(base_val, &var_path.fields, expr.span);
                sub_val.clone()
            }
            Expr::Lit(lit) => match lit {
                Literal::Int(val) => Val::Int(*val),
                Literal::Str(val) => Val::Str(*val),
                Literal::Bool(val) => Val::Bool(*val),
                Literal::Empty(val) => Val::Empty(*val),
            },
        }
    }

    fn eval_resolve_var_path<'e, 'v>(
        &mut self,
        var: &'v mut Val<'e, 'p>,
        path: &[&'p str],
        span: Span,
    ) -> &'v mut Val<'e, 'p> {
        let mut cur_val = var;
        'path: for field in path {
            let temp = cur_val;
            match temp {
                Val::Tuple(args) => {
                    if let Some(field_val) = field
                        .parse::<usize>()
                        .ok()
                        .and_then(|idx| args.get_mut(idx))
                    {
                        cur_val = field_val;
                        continue 'path;
                    } else {
                        let val = "TODO"; // self.format_val(cur_val, true, 0);
                        self.error(
                            format!("Runtime Error: {} is not a valid index of {}", field, val,),
                            span,
                        )
                    }
                }
                Val::Struct(_name, fields) => {
                    for (field_name, field_val) in fields {
                        if field_name == field {
                            cur_val = field_val;
                            continue 'path;
                        }
                    }
                    let val = "TODO"; // self.format_val(cur_val, true, 0);
                    self.error(
                        format!("Runtime Error: {} is not a valid index of {}", field, val,),
                        span,
                    )
                }
                _ => {
                    let val = "TODO"; // self.format_val(cur_val, true, 0);
                    self.error(
                        format!(
                            "Runtime Error: Tried to get a field on non-composite {}",
                            val,
                        ),
                        span,
                    )
                }
            }
        }
        cur_val
    }

    fn eval_resolve_var<'e, 'v>(
        &mut self,
        var: &'p str,
        envs: &'v mut Vec<Env<'e, 'p>>,
    ) -> &'v mut Val<'e, 'p> {
        for env in envs.iter_mut().rev() {
            if let Some(val) = env.vals.get_mut(var) {
                return val;
            }
        }
        self.error(
            format!("Runtime Error: Use of undefined var {}", var),
            self.cur_eval_span,
        )
    }

    fn print_val<'e>(&mut self, val: &Val<'e, 'p>) {
        let string = self.format_val(val, false, 0);
        println!("{}", string);

        if let Some(output) = self.output.as_mut() {
            output.push_str(&string);
            output.push_str("\n");
        }
    }

    fn format_val<'e>(&mut self, val: &Val<'e, 'p>, debug: bool, indent: usize) -> String {
        match val {
            Val::Int(int) => {
                format!("{}", int)
            }
            Val::Str(string) => {
                if debug {
                    format!(r#""{}""#, string)
                } else {
                    format!("{}", string)
                }
            }
            Val::Bool(boolean) => {
                format!("{}", boolean)
            }
            Val::Empty(_) => {
                format!("()")
            }
            Val::Tuple(tuple) => {
                let mut f = String::new();
                write!(f, "(").unwrap();
                for (idx, val) in tuple.iter().enumerate() {
                    if idx != 0 {
                        write!(f, ", ").unwrap();
                    }
                    // Debug print fields unconditionally to make 0 vs "0" clear
                    let val = self.format_val(val, true, indent);
                    write!(f, "{}", val).unwrap();
                }
                write!(f, ")").unwrap();
                f
            }
            Val::Struct(name, args) => {
                let mut f = String::new();
                write!(f, "{} {{ ", name).unwrap();
                for (idx, (field, val)) in args.iter().enumerate() {
                    if idx != 0 {
                        write!(f, ", ").unwrap();
                    }
                    // Debug print fields unconditionally to make 0 vs "0" clear
                    let val = self.format_val(val, true, indent);
                    write!(f, "{}: {}", field, val).unwrap();
                }
                write!(f, " }}").unwrap();
                f
            }
            Val::Func(closure) => {
                let mut f = String::new();
                write!(f, "fn {}(", closure.func.name).unwrap();
                for (i, arg) in closure.func.args.iter().enumerate() {
                    if i != 0 {
                        write!(f, ", ").unwrap();
                    }
                    write!(f, "{}", arg.ident).unwrap();
                }
                writeln!(f, ")").unwrap();

                if !closure.captures.is_empty() {
                    let indent = indent + 2;
                    write!(f, "{:indent$}captures:", "", indent = indent).unwrap();
                    for (arg, capture) in &closure.captures {
                        writeln!(f, "").unwrap();
                        let sub_indent = indent + arg.len() + 2;
                        // Debug print captures unconditionally to make 0 vs "0" clear
                        let val = self.format_val(capture, true, sub_indent);
                        write!(f, "{:indent$}- {}: {}", "", arg, val, indent = indent).unwrap();
                    }
                }
                f
            }
            Val::Builtin(builtin) => {
                let mut f = String::new();
                write!(f, "builtin {}(", builtin.name).unwrap();
                for (i, arg) in builtin.args.iter().enumerate() {
                    if i != 0 {
                        write!(f, ", ").unwrap();
                    }
                    write!(f, "{}", arg).unwrap();
                }
                writeln!(f, ")").unwrap();
                f
            }
        }
    }
}
