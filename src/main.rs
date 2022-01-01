use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::{self, Write};

use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{alpha1, alphanumeric1, char, space0, space1},
    combinator::{map, recognize, rest},
    error::ParseError,
    multi::{many0, separated_list0, separated_list1},
    sequence::pair,
    AsChar, IResult, InputTakeAtPosition, Parser,
};

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
    struct Point {
        x: Int,
        y: Int,
        z: Int,
    }

    fn handle_point(point: Point) -> Point {
        print "innocent"
        print point
        ret point
    }

    fn handle_pointer(ptr: fn(Point) -> Point, pt: Point) -> Int {
        let pt2: Point = ptr(pt)
        print pt2
        ret 0
    }

    let f: fn(Point) -> Point = handle_point

    let _: Int = handle_pointer(f, Point { x: 1, y: 3, z: 7 })

    struct Point {
        x: Int,
        y: Int,
    }

    fn handle_point_2(point: Point) -> Point {
        print "evil"
        print point
        ret point
    }

    set f = handle_point_2

    let _: Int = handle_pointer(f, Point { x: 2, y: 5 })

    ret 0
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
            eprintln!("{}", line,);
            for _ in 0..start_col {
                eprint!(" ");
            }
            for _ in start_col..end_col {
                eprint!("~");
            }
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
    ty: Ty<'p>,
    captures: HashSet<&'p str>,
}

#[derive(Debug, Clone)]
struct Struct<'p> {
    name: &'p str,
    fields: Vec<FieldDecl<'p>>,
}

#[derive(Debug, Clone)]
struct Statement<'p> {
    code: Stmt<'p>,
    span: Span,
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
        name: &'p str,
        expr: Expression<'p>,
    },
    Func {
        func: Function<'p>,
    },
    Struct(Struct<'p>),
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
    Var(&'p str),
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
    fn ty(&self) -> Ty<'static> {
        match self {
            Literal::Int(_) => Ty::Int,
            Literal::Str(_) => Ty::Str,
            Literal::Bool(_) => Ty::Bool,
            Literal::Empty(_) => Ty::Empty,
        }
    }
}

#[derive(Debug, Clone)]
struct VarDecl<'p> {
    ident: &'p str,
    ty: Ty<'p>,
}
#[derive(Debug, Clone)]
struct FieldDecl<'p> {
    ident: &'p str,
    ty: Ty<'p>,
}

// Parse intermediates
enum Item<'p> {
    Comment(&'p str),
    Struct(&'p str),
    Func(&'p str, Vec<VarDecl<'p>>, Ty<'p>),
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
    ty: Ty<'static>,
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
            ty: Ty::Func {
                arg_tys: vec![],
                return_ty: Box::new(Ty::Int),
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

                    Stmt::Struct(Struct { name, fields })
                }
                Item::Func(name, args, return_ty) => {
                    let (new_i, (Block(block_stmts), terminal)) = self.parse_block(i)?;
                    i = new_i;

                    if !matches!(terminal, Item::End) {
                        self.error(
                            format!("Parse Error: `fn` ending eith an `else`"),
                            Span {
                                start: addr(i),
                                end: addr(i),
                            },
                        )
                    }

                    Stmt::Func {
                        func: Function {
                            ty: Ty::Func {
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
                                format!("Parse Error: `else` ending eith an `else`"),
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
                            format!("Parse Error: `loop` ending eith an `else`"),
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
        (i, Ty::Unknown)
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

fn return_ty(i: &str) -> IResult<&str, Ty> {
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
    let (i, name) = ident(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag("=")(i)?;
    let (i, _) = space0(i)?;
    let (i, expr) = expr(i)?;

    Ok((i, Stmt::Set { name, expr }))
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
    let (i, expr) = alt((expr_tuple, expr_named, expr_call, expr_lit, expr_var))(i)?;
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

fn expr_var(i: &str) -> IResult<&str, Expr> {
    map(ident, Expr::Var)(i)
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
    alt((
        typed_ident,
        map(ident, |id| VarDecl {
            ident: id,
            ty: Ty::Unknown,
        }),
    ))(i)
}

fn ident(i: &str) -> IResult<&str, &str> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ))(i)
}

fn typed_ident(i: &str) -> IResult<&str, VarDecl> {
    let (i, id) = ident(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag(":")(i)?;
    let (i, _) = space0(i)?;
    let (i, ty) = ty_ref(i)?;

    Ok((i, VarDecl { ident: id, ty }))
}

fn ty_ref(i: &str) -> IResult<&str, Ty> {
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

fn ty_ref_int(i: &str) -> IResult<&str, Ty> {
    map(tag("Int"), |_| Ty::Int)(i)
}
fn ty_ref_str(i: &str) -> IResult<&str, Ty> {
    map(tag("Str"), |_| Ty::Str)(i)
}
fn ty_ref_bool(i: &str) -> IResult<&str, Ty> {
    map(tag("Bool"), |_| Ty::Bool)(i)
}
fn ty_ref_empty(i: &str) -> IResult<&str, Ty> {
    map(tag("()"), |_| Ty::Empty)(i)
}
fn ty_ref_func(i: &str) -> IResult<&str, Ty> {
    let (i, _) = tag("fn")(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag("(")(i)?;
    let (i, _) = space0(i)?;
    let (i, arg_tys) = separated_list0(char(','), padded(ty_ref))(i)?;
    let (i, _) = tag(")")(i)?;
    let (i, return_ty) = return_ty(i)?;

    Ok((
        i,
        Ty::Func {
            arg_tys,
            return_ty: Box::new(return_ty),
        },
    ))
}
fn ty_ref_tuple(i: &str) -> IResult<&str, Ty> {
    let (i, _) = tag("(")(i)?;
    let (i, _) = space0(i)?;
    let (i, tys) = separated_list1(char(','), padded(ty_ref))(i)?;
    let (i, _) = tag(")")(i)?;

    Ok((i, Ty::Tuple(tys)))
}
fn ty_ref_named(i: &str) -> IResult<&str, Ty> {
    map(ident, |name| Ty::Named(name))(i)
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Ty<'p> {
    Int,
    Str,
    Bool,
    Empty,
    Func {
        arg_tys: Vec<Ty<'p>>,
        return_ty: Box<Ty<'p>>,
    },
    Tuple(Vec<Ty<'p>>),
    Named(&'p str),
    Unknown,
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
}

/// Information about types for a specific scope.
struct CheckEnv<'p> {
    /// The types of variables
    vars: HashMap<&'p str, TyIdx>,
    /// The struct definitions and TyIdx's
    tys: HashMap<&'p str, (TyIdx, Struct<'p>)>,

    /// Whether this scope is the root of a function
    /// (the scope of its arguments). If you walk over
    /// this kind of frame, then you're accessing captures.
    is_function_root: bool,
}
struct CheckEntry<'a, 'p> {
    is_local: bool,
    entry: std::collections::hash_map::OccupiedEntry<'a, &'p str, TyIdx>,
}

impl<'p> TyCtx<'p> {
    fn resolve_var<'a>(&'a mut self, var_name: &'p str) -> Option<CheckEntry<'a, 'p>> {
        // By default we're accessing locals
        let mut is_local = true;
        use std::collections::hash_map::Entry;
        for env in self.envs.iter_mut().rev() {
            if let Entry::Occupied(entry) = env.vars.entry(var_name) {
                return Some(CheckEntry { is_local, entry });
            }
            if env.is_function_root {
                // We're walking over a function root, so we're now
                // accessing captures.
                is_local = false;
            }
        }
        None
    }

    fn push_struct_decl(&mut self, struct_decl: Struct<'p>) -> TyIdx {
        let ty_idx = self.tys.len();
        self.tys.push(Ty::Named(struct_decl.name));
        self.envs
            .last_mut()
            .unwrap()
            .tys
            .insert(struct_decl.name, (ty_idx, struct_decl));
        ty_idx
    }

    fn resolve_nominal_ty<'a>(&'a mut self, ty_name: &'p str) -> Option<&'a (TyIdx, Struct<'p>)> {
        if self.is_typed {
            for (_depth, env) in self.envs.iter_mut().rev().enumerate() {
                if let Some(ty) = env.tys.get(ty_name) {
                    return Some(ty);
                }
            }
            None
        } else {
            None
        }
    }

    fn memoize_ty(&mut self, program: &mut Program<'p>, ty: &Ty<'p>) -> TyIdx {
        if self.is_typed {
            if let Ty::Named(name) = ty {
                if let Some((idx, _)) = self.resolve_nominal_ty(name) {
                    *idx
                } else {
                    // TODO: rejig this so the line info is better
                    program.error(
                        format!("Compile Error: use of undefined type name: {}", name),
                        Span {
                            start: addr(program.input),
                            end: addr(program.input),
                        },
                    )
                }
            } else if let Some(idx) = self.ty_map.get(ty) {
                *idx
            } else {
                let ty1 = ty.clone();
                let ty2 = ty.clone();
                let idx = self.tys.len();
                self.ty_map.insert(ty1, idx);
                self.tys.push(ty2);
                idx
            }
        } else {
            0
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
}

impl<'p> Program<'p> {
    fn check(&mut self) {
        let mut ctx = TyCtx {
            tys: Vec::new(),
            ty_map: HashMap::new(),
            envs: Vec::new(),
            is_typed: self.typed,
        };

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

        let mut main = self.main.take().unwrap();
        self.check_func(&mut main, &mut ctx);

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

    fn check_func(&mut self, func: &mut Function<'p>, ctx: &mut TyCtx<'p>) {
        let vars = func
            .args
            .iter()
            .map(|decl| (decl.ident, ctx.memoize_ty(self, &decl.ty)))
            .collect();

        ctx.envs.push(CheckEnv {
            vars,
            tys: HashMap::new(),
            is_function_root: true,
        });

        let mut captures = HashSet::new();

        let return_ty = if let Ty::Func { return_ty, .. } = &func.ty {
            &*return_ty
        } else {
            &Ty::Unknown
        };
        let return_ty = ctx.memoize_ty(self, return_ty);

        self.check_block(&mut func.stmts, ctx, &mut captures, return_ty);

        func.captures = captures;
        ctx.envs.pop();
    }

    fn check_block(
        &mut self,
        stmts: &mut [Statement<'p>],
        ctx: &mut TyCtx<'p>,
        captures: &mut HashSet<&'p str>,
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
                    let expected_ty = ctx.memoize_ty(self, &Ty::Bool);

                    self.check_ty(ctx, expr_ty, expected_ty, "`if`", expr.span);
                    self.check_block(stmts, ctx, captures, return_ty);
                    self.check_block(else_stmts, ctx, captures, return_ty);
                }
                Stmt::Loop { stmts } => {
                    self.check_block(stmts, ctx, captures, return_ty);
                }
                Stmt::Struct(struct_decl) => {
                    let ty_idx = ctx.push_struct_decl(struct_decl.clone());
                    ctx.envs
                        .last_mut()
                        .unwrap()
                        .tys
                        .insert(struct_decl.name, (ty_idx, struct_decl.clone()));
                }
                Stmt::Func { func } => {
                    // We push a func's name after checking it to avoid
                    // infinite capture recursion. This means naive recursion
                    // is illegal.
                    self.check_func(func, ctx);
                    let func_ty = ctx.memoize_ty(self, &func.ty);
                    ctx.envs.last_mut().unwrap().vars.insert(func.name, func_ty);
                }
                Stmt::Let { name, expr } => {
                    let expr_ty = self.check_expr(expr, ctx, captures);
                    let expected_ty = ctx.memoize_ty(self, &name.ty);

                    self.check_ty(ctx, expr_ty, expected_ty, "`let`", expr.span);

                    ctx.envs
                        .last_mut()
                        .unwrap()
                        .vars
                        .insert(name.ident, expr_ty);
                }
                Stmt::Set { name, expr } => {
                    let expr_ty = self.check_expr(expr, ctx, captures);

                    if let Some(mut var) = ctx.resolve_var(name) {
                        if var.is_local {
                            let expected_ty = *var.entry.get();
                            var.entry.insert(expr_ty);
                            self.check_ty(ctx, expr_ty, expected_ty, "`set`", expr.span);
                        } else {
                            self.error(
                                format!("Compile Error: Trying to `set` captured variable '{}' (captures are by-value!)", name),
                                *stmt_span,
                            )
                        }
                    } else {
                        self.error(
                            format!(
                                "Compile Error: Trying to `set` undefined variable '{}'",
                                name
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
        captures: &mut HashSet<&'p str>,
    ) -> TyIdx {
        match &expr.code {
            Expr::Lit(lit) => {
                return ctx.memoize_ty(self, &lit.ty());
            }
            Expr::Var(var_name) => {
                if let Some(var) = ctx.resolve_var(var_name) {
                    if !var.is_local {
                        captures.insert(var_name);
                    }
                    return *var.entry.get();
                } else {
                    self.error(
                        format!("Compile Error: Use of undefined variable '{}'", var_name),
                        expr.span,
                    )
                }
            }
            Expr::Tuple(args) => {
                let arg_tys = args
                    .iter()
                    .map(|arg| {
                        let ty = self.check_expr(arg, ctx, captures);
                        ctx.realize_ty(ty).clone()
                    })
                    .collect();
                ctx.memoize_ty(self, &Ty::Tuple(arg_tys))
            }
            Expr::Named { name, args } => {
                let query = ctx.resolve_nominal_ty(name).cloned();
                if let Some((ty_idx, ty_decl)) = query {
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
                        let expected_ty = ctx.memoize_ty(self, &field_decl.ty);
                        self.check_ty(ctx, expr_ty, expected_ty, "struct literal", arg.span);
                    }
                    ty_idx
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
                    return ctx.memoize_ty(self, &Ty::Named(name));
                }
            }
            Expr::Call { func, args } => {
                if let Some(var) = ctx.resolve_var(func) {
                    if !var.is_local {
                        captures.insert(func);
                    }

                    let var_ty = *var.entry.get();
                    let func_ty = ctx.realize_ty(var_ty).clone();
                    let (arg_tys, return_ty) = if let Ty::Func { arg_tys, return_ty } = func_ty {
                        let arg_tys = arg_tys.clone();
                        let return_ty = return_ty.clone();
                        (
                            arg_tys.iter().map(|ty| ctx.memoize_ty(self, ty)).collect(),
                            ctx.memoize_ty(self, &return_ty),
                        )
                    } else if self.typed {
                        self.error(
                            format!("Compile Error: Function call must have Func type!"),
                            expr.span,
                        )
                    } else {
                        (Vec::new(), ctx.memoize_ty(self, &Ty::Unknown))
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
                            .unwrap_or(ctx.memoize_ty(self, &Ty::Unknown));

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
            let msg = format!(
                "Compile Error: {} type mismatch (expected {:?}, got {:?})",
                env_name,
                ctx.realize_ty(expected_ty),
                ctx.realize_ty(computed_ty),
            );
            self.error(msg, span)
        }
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
            ty: Ty::Func {
                arg_tys: vec![Ty::Int, Ty::Int],
                return_ty: Box::new(Ty::Int),
            },
            func: builtin_add,
        },
        Builtin {
            name: "sub",
            args: &["lhs", "rhs"],
            ty: Ty::Func {
                arg_tys: vec![Ty::Int, Ty::Int],
                return_ty: Box::new(Ty::Int),
            },
            func: builtin_sub,
        },
        Builtin {
            name: "mul",
            args: &["lhs", "rhs"],
            ty: Ty::Func {
                arg_tys: vec![Ty::Int, Ty::Int],
                return_ty: Box::new(Ty::Int),
            },
            func: builtin_mul,
        },
        Builtin {
            name: "eq",
            args: &["lhs", "rhs"],
            ty: Ty::Func {
                arg_tys: vec![Ty::Int, Ty::Int],
                return_ty: Box::new(Ty::Bool),
            },
            func: builtin_eq,
        },
        Builtin {
            name: "not",
            args: &["rhs"],
            ty: Ty::Func {
                arg_tys: vec![Ty::Bool],
                return_ty: Box::new(Ty::Bool),
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
    fn ty(&self) -> &Ty {
        match self {
            Val::Int(_) => &Ty::Int,
            Val::Str(_) => &Ty::Str,
            Val::Bool(_) => &Ty::Bool,
            Val::Empty(_) => &Ty::Empty,
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
                Stmt::Set { name, expr } => {
                    let val = self.eval_expr(expr, envs);

                    let mut found = false;
                    for env in envs.iter_mut().rev() {
                        if let Some(_) = env.vals.get(name) {
                            found = true;
                            env.vals.insert(*name, val).unwrap();
                            break;
                        }
                    }

                    if !found {
                        self.error(
                            format!(
                                "Runtime Error: Tried to set an undefined local variable {}",
                                name
                            ),
                            *stmt_span,
                        )
                    }
                }
                Stmt::Struct(_struct_decl) => {
                    // TODO: ?
                }
                Stmt::Func { func } => {
                    let captures = func
                        .captures
                        .iter()
                        .map(|&var| (var, self.eval_resolve_var(var, envs)))
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
                let func = self.eval_resolve_var(func_name, envs);
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
            Expr::Var(var) => self.eval_resolve_var(var, envs),
            Expr::Lit(lit) => match lit {
                Literal::Int(val) => Val::Int(*val),
                Literal::Str(val) => Val::Str(*val),
                Literal::Bool(val) => Val::Bool(*val),
                Literal::Empty(val) => Val::Empty(*val),
            },
        }
    }

    fn eval_resolve_var<'e>(&mut self, var: &'p str, envs: &mut Vec<Env<'e, 'p>>) -> Val<'e, 'p> {
        for env in envs.iter().rev() {
            if let Some(val) = env.vals.get(var) {
                return val.clone();
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
// tests!
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
//

#[cfg(test)]
mod test {
    fn run(input: &str) -> (i64, Option<String>) {
        crate::Program::untyped(input).run()
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_set_undefined() {
        let program = r#"
            set x = 1
            let x = 0
            ret x
        "#;

        let (_result, _output) = run(program);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_set_capture() {
        let program = r#"
            let x = 0
            fn f() {
                set x = 1
                ret x
            }
            ret f()
        "#;

        let (_result, _output) = run(program);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_recursive() {
        let program = r#"
            fn recursive() {
                if false {
                    // Can't self-reference
                    ret recursive()
                } else {
                    ret 0
                }
            }

            ret recursive()
        "#;

        let (_result, _output) = run(program);
    }

    #[test]
    #[should_panic(expected = "Parse Error")]
    fn parse_fail_basic() {
        let program = r#"
            !
        "#;

        let (_result, _output) = run(program);
    }

    #[test]
    #[should_panic(expected = "Parse Error")]
    fn parse_fail_else_func() {
        let program = r#"
            fn f() {
                ret 0
            } else {
                print 1
            }
            
            ret f()
        "#;

        let (_result, _output) = run(program);
    }

    #[test]
    #[should_panic(expected = "Parse Error")]
    fn parse_fail_else_loop() {
        let program = r#"
            loop {
                break
            } else {
                print "oh no"
            }
            
            ret 0
        "#;

        let (_result, _output) = run(program);
    }

    #[test]
    #[should_panic(expected = "Parse Error")]
    fn parse_fail_else_else() {
        let program = r#"
            if true {
                ret 0
            } else {
                ret 1
            } else {
                ret 2
            }
            
            ret 3
        "#;

        let (_result, _output) = run(program);
    }

    #[test]
    #[should_panic(expected = "Runtime Error")]
    fn eval_fail_if_type() {
        let program = r#"
            if 0 {
                ret 0
            }
            ret 1
        "#;

        let (_result, _output) = run(program);
    }

    #[test]
    #[should_panic(expected = "Runtime Error")]
    fn eval_fail_add_count() {
        let program = r#"
            ret add(0, 1, 2)
        "#;

        let (_result, _output) = run(program);
    }

    #[test]
    #[should_panic(expected = "Runtime Error")]
    fn eval_fail_add_types() {
        let program = r#"
            ret add(true, false)
        "#;

        let (_result, _output) = run(program);
    }

    #[test]
    #[should_panic(expected = "Runtime Error")]
    fn eval_fail_not_count() {
        let program = r#"
            let x = not(true, false)
            ret 0
        "#;

        let (_result, _output) = run(program);
    }

    #[test]
    #[should_panic(expected = "Runtime Error")]
    fn eval_fail_not_type() {
        let program = r#"
            let x = not(0)
            ret 0
        "#;

        let (_result, _output) = run(program);
    }

    #[test]
    #[should_panic(expected = "Runtime Error")]
    fn eval_fail_eq_count() {
        let program = r#"
            let x = eq(1, 2, 3)
            ret 0
        "#;

        let (_result, _output) = run(program);
    }

    #[test]
    #[should_panic(expected = "Runtime Error")]
    fn eval_fail_eq_type() {
        let program = r#"
            let x = eq(true, false)
            ret 0
        "#;

        let (_result, _output) = run(program);
    }

    #[test]
    #[should_panic(expected = "Runtime Error")]
    fn eval_fail_call_type() {
        let program = r#"
            let x = 0
            ret x()
        "#;

        let (_result, _output) = run(program);
    }

    #[test]
    #[should_panic(expected = "Runtime Error")]
    fn eval_fail_no_ret() {
        let program = r#"
            fn f() {
                print "hello"
            }
            ret f()
        "#;

        let (_result, _output) = run(program);
    }

    #[test]
    #[should_panic(expected = "Runtime Error")]
    fn eval_fail_break_no_loop() {
        let program = r#"
            continue
            ret 0
        "#;

        let (_result, _output) = run(program);
    }

    #[test]
    #[should_panic(expected = "Runtime Error")]
    fn eval_fail_continue_no_loop() {
        let program = r#"
            break
            ret 0
        "#;

        let (_result, _output) = run(program);
    }

    #[test]
    #[should_panic(expected = "Runtime Error")]
    fn eval_fail_too_many_args() {
        let program = r#"
            fn f() {
                ret 0
            }
            ret f(1)
        "#;

        let (_result, _output) = run(program);
    }

    #[test]
    #[should_panic(expected = "Runtime Error")]
    fn eval_fail_too_few_args() {
        let program = r#"
            fn f(x, y) {
                ret 0
            }
            ret f(1)
        "#;

        let (_result, _output) = run(program);
    }

    #[test]
    #[should_panic(expected = "Runtime Error")]
    fn eval_fail_dupe_args() {
        let program = r#"
            // Can't give args the same name!
            fn f(x, x) {
                ret 0
            }
            ret f(1)
        "#;

        let (_result, _output) = run(program);
    }

    #[test]
    fn test_factorial() {
        let program = r#"
            fn factorial(self, val) {
                if eq(val, 0) {
                    ret 1
                }
                ret mul(self(self, sub(val, 1)), val)
            }

            print factorial(factorial, 0)
            print factorial(factorial, 1)
            print factorial(factorial, 2)
            print factorial(factorial, 3)
            print factorial(factorial, 4)
            print factorial(factorial, 5)
            print factorial(factorial, 6)
            print factorial(factorial, 7)

            ret 0
        "#;

        let (result, output) = run(program);
        assert_eq!(result, 0);
        assert_eq!(
            output.unwrap(),
            r#"1
1
2
6
24
120
720
5040
"#
        )
    }

    #[test]
    fn test_comments() {
        let program = r#"
            // Hello!
            let x = 0
            // let x = 2
            // print "fuck!"
            print "yay"
            
            let whatever = 9

            // fn whatever() {
            //   ret 7
            // }
            //

            // }

            print whatever

            // ret -1
            ret x
        "#;

        let (result, output) = run(program);
        assert_eq!(result, 0);
        assert_eq!(
            output.unwrap(),
            r#"yay
9
"#
        );
    }

    #[test]
    fn test_if() {
        let program = r#"
            let x = true
            let y = 2

            fn captures() {
                if x {
                    ret y
                }
            }

            fn False() {
                ret false
            }

            if x {
                print "yes1"
                print y
            }
            
            print "normal1"
            
            if False() {
                print "oh no!"
                ret -2
            } else {
                print "else1"
            }

            if False() {
                print "oh no!"
            }

            print "normal2"

            let x = false
            let y = 3
            print captures()
            print x
            print y

            fn captures2() {
                if x {
                    ret 1
                } else {
                    ret sub(y, 4)
                }
            }

            let x = 1
            let y = 4
            print captures2()
            print x
            print y

            if true {
                print "yes2"
                ret 999
            }

            ret -1
        "#;

        let (result, output) = run(program);
        assert_eq!(result, 999);
        assert_eq!(
            output.unwrap(),
            r#"yes1
2
normal1
else1
normal2
2
false
3
-1
1
4
yes2
"#
        );
    }

    #[test]
    fn test_builtin_bool() {
        let program = r#"
            let x = 0
            if eq(x, 0) {
                print "eq!"
            }
            if not(eq(x, 1)) {
                print "neq!"
            }

            if eq(x, 1) {
                ret -1
            }
            ret 0
        "#;

        let (result, output) = run(program);
        assert_eq!(result, 0);
        assert_eq!(
            output.unwrap(),
            r#"eq!
neq!
"#
        );
    }

    #[test]
    fn test_builtin_math() {
        let program = r#"
            ret sub(mul(add(4, 7), 13), 9)
        "#;

        let (result, _output) = run(program);
        assert_eq!(result, (4 + 7) * 13 - 9);
    }

    #[test]
    fn test_first_class_basic() {
        let program = r#"
            let capture = 123
            fn doit() {
                ret capture
            }
            fn higher(func) {
                ret func()
            }

            ret higher(doit)
        "#;

        let (result, _output) = run(program);
        assert_eq!(result, 123);
    }

    #[test]
    fn test_first_class_with_args() {
        let program = r#"
            let capture = 123
            fn doit(x, y) {
                ret x
            }
            fn higher(func) {
                ret func(777, 999)
            }

            ret higher(doit)
        "#;

        let (result, _output) = run(program);
        assert_eq!(result, 777);
    }

    #[test]
    fn test_first_class_with_captures() {
        let program = r#"
            let capture = 666
            fn capturedfunc() {
                ret capture
            }
            fn doit(x, y) {
                ret capturedfunc()
            }
            fn higher(func) {
                ret func(777, 999)
            }

            ret higher(doit)
        "#;

        let (result, _output) = run(program);
        assert_eq!(result, 666);
    }

    #[test]
    fn test_captures() {
        let program = r#"
            let shadowed = 66
            fn captures() {
                ret shadowed
            }
            fn mask(shadowed) {
                ret captures()
            }

            ret mask(33)
        "#;

        let (result, _output) = run(program);
        assert_eq!(result, 66);
    }

    #[test]
    fn test_call_capture() {
        let program = r#"
            fn double(x) {
                fn two(val) {
                    ret x(x(val))
                }
                ret two
            }

            fn succ(x) {
                ret add(x, 1)
            }

            let add_two = double(succ)
            let add_four = double(add_two)
            let add_eight = double(add_four)

            let a = add_two(1)
            let b = add_four(1)
            let c = add_eight(1)

            ret add(add(a, b), c)
        "#;

        let (result, _output) = run(program);
        assert_eq!(result, 17);
    }

    #[test]
    fn test_negative() {
        let program = r#"
            ret -1
        }
        "#;

        let (result, _output) = run(program);
        assert_eq!(result, -1);
    }

    #[test]
    fn test_fake_bools() {
        let program = r#"
            fn True(if, else) {
                ret if()
            }
            fn False(if, else) {
                ret else()
            }

            let condition = True
            let capture = 69

            fn printTrue() {
                print 1
                ret add(capture, 1)
            }
            fn printFalse() {
                print 0
                ret add(capture, 0)
            }

            ret condition(printTrue, printFalse)
        "#;

        let (result, output) = run(program);
        assert_eq!(result, 70);
        assert_eq!(output.unwrap(), "1\n");
    }

    #[test]
    fn test_idents() {
        let program = r#"
            let _x = 66
            let __y = 55
            let _0 = 44
            let _x_y__z_ = 33
            ret add(add(add(_x, __y), _0), _x_y__z_)
        "#;

        let (result, _output) = run(program);
        assert_eq!(result, 66 + 55 + 44 + 33);
    }

    #[test]
    fn test_literals() {
        let program = r#"
            let a = 66
            let b = -55
            let c = "hello"
            let d = ""
            let e = true
            let f = false
            let g = ()

            print a
            print b
            print c
            print d
            print e
            print f
            print g

            ret a
        "#;

        let (result, output) = run(program);
        assert_eq!(result, 66);
        assert_eq!(
            output.unwrap(),
            r#"66
-55
hello

true
false
()
"#
        );
    }

    #[test]
    fn test_str() {
        let program = r#"
            print "hello"
            ret 1
        "#;

        let (result, output) = run(program);
        assert_eq!(result, 1);
        assert_eq!(output.unwrap(), "hello\n");
    }

    #[test]
    fn test_basic() {
        // Just tests basic functionality.
        //
        // Whitespace is wonky to make sure the parser is pemissive of that.
        let program = r#"
            fn x(a, b , c ) {
                fn sub (e , f){
                    print c
                    ret f
                } 
                ret sub( a, b)
            }

            fn  y(z ) {
                ret z
            }

            fn g( ) { 
                fn h() {
                    ret 2 
                }
                ret h()
            }



            let val1=g( )
            let val2  = x(1, 99, 12)
            let  val3= 21

            let result = x(7, x(val3 , y(x(2, val2, 7)), 8 ), 1)

            ret result
        "#;

        let (result, _output) = run(program);
        assert_eq!(result, 99);
    }

    #[test]
    fn test_loops_no_looping() {
        let program = r#"
            loop {
                if true {
                    print "yay1"
                    break
                } else {
                    continue
                }
            }

            let x = 17
            fn do_stuff() {
                loop {
                    print x
                    ret 2
                }
            }

            let x = 12
            ret do_stuff()
        "#;

        let (result, output) = run(program);
        assert_eq!(result, 2);
        assert_eq!(
            output.unwrap(),
            r#"yay1
17
"#
        );
    }

    #[test]
    fn test_loops_and_set() {
        let program = r#"
            let x = 10

            fn remembers_original() {
                ret x
            }
            loop {
                fn remembers_previous() {
                    ret x
                }

                // Exit the loop at 0
                if eq(x, 0) {
                    break
                }
                set x = sub(x, 1)

                // Skip 2, no one likes 2
                if eq(x, 2) {
                    continue
                }

                print "loop!"
                print remembers_original()
                print remembers_previous()
                print x
            }

            ret 0
        "#;

        let (result, output) = run(program);
        assert_eq!(result, 0);
        assert_eq!(
            output.unwrap(),
            r#"loop!
10
10
9
loop!
10
9
8
loop!
10
8
7
loop!
10
7
6
loop!
10
6
5
loop!
10
5
4
loop!
10
4
3
loop!
10
2
1
loop!
10
1
0
"#
        );
    }

    #[test]
    fn test_set_basic() {
        let program = r#"
            let x = 0
            print x
            
            set x = 3
            print x
            
            set x = add(x, 8)
            print x
            
            if true {
                set x = 27
            } else {
                set x = 4
            }
            print x

            if false {
                set x = 2
            } else {
                set x = 35
            }
            print x

            // reinitialize value
            let x = 58
            print x

            set x = 71
            print x

            ret 0
        "#;

        let (result, output) = run(program);
        assert_eq!(result, 0);
        assert_eq!(
            output.unwrap(),
            r#"0
3
11
27
35
58
71
"#
        );
    }

    #[test]
    fn test_aggregates_basic() {
        let program = r#"
            let factors: (Int, Bool) = (0, true)
            print factors
            set factors = (2, false)
            print factors

            struct Point {
                x: Int
                y: Int
            }

            let pt: Point = Point { x: 0, y: 1 }
            print pt
            set pt = Point { x: 3, y: 4 }
            print pt

            ret 0
        "#;

        let (result, output) = run(program);
        assert_eq!(result, 0);
        assert_eq!(
            output.unwrap(),
            r#"(0, true)
(2, false)
Point { x: 0, y: 1 }
Point { x: 3, y: 4 }
"#
        )
    }

    #[test]
    fn test_aggregates_anon_structs() {
        // In untyped mode, we can use structs without declaring them,
        // and change the type of aggregates as we please.
        let program = r#"
            let factors = (0, true)
            print factors
            set factors = (2, "bye", ())
            print factors

            let pt = Point { x: 0, y: 1 }
            print pt
            set pt = Point { x: 3, y: true, z: "hello" }
            print pt

            ret 0
        "#;

        let (result, output) = run(program);
        assert_eq!(result, 0);
        assert_eq!(
            output.unwrap(),
            r#"(0, true)
(2, "bye", ())
Point { x: 0, y: 1 }
Point { x: 3, y: true, z: "hello" }
"#
        )
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
//
//
//

#[cfg(test)]
mod test_typed {
    fn run_typed(input: &str) -> (i64, Option<String>) {
        crate::Program::typed(input).run()
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_evil_nominal_smuggling_complex() {
        // If you don't handle nested nominal types deeply, then you can get
        // situations where the type changes from underneath you!
        let program = r#"
            struct Point {
                x: Int,
                y: Int,
                z: Int,
            }

            fn handle_point(point: Point) -> Point {
                print "innocent"
                print point
                ret point
            }

            fn handle_pointer(ptr: fn(Point) -> Point, pt: Point) -> Int {
                let pt2: Point = ptr(pt)
                print pt2
                ret 0
            }

            let f: fn(Point) -> Point = handle_point

            let _: Int = handle_pointer(f, Point { x: 1, y: 3, z: 7 })

            struct Point {
                x: Int,
                y: Int,
            }

            fn handle_point_2(point: Point) -> Point {
                print "evil"
                print point
                ret point
            }

            set f = handle_point_2

            let _: Int = handle_pointer(f, Point { x: 2, y: 5 })

            ret 0
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 0);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_evil_nominal_smuggling_simple_1() {
        // If you don't handle shadowed nominal types, then you can get
        // situations where the type changes from underneath you!
        let program = r#"
            struct Point {
                x: Int,
                y: Int,
                z: Int,
            }

            let x: Point = Point { x: 1, y: 3, z: 7 }
            print x

            struct Point {
                x: Int,
                y: Int,
            }
            set x = Point { x: 2, y: 5 }
            print x

            ret 0
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 0);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_evil_nominal_smuggling_simple_2() {
        // If you don't handle nested nominal types deeply, then you can get
        // situations where the type changes from underneath you!
        let program = r#"
            struct Point {
                x: Int,
                y: Int,
                z: Int,
            }

            let x: (Point, Point) = (Point { x: 1, y: 3, z: 7 }, Point { x: 2, y: 5, z: 8})
            print x

            struct Point {
                x: Int,
                y: Int,
            }
            set x = (Point { x: 2, y: 5 }, Point { x: 3, y: 4 })
            print x

            ret 0
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 0);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_evil_nominal_smuggling_simple_3() {
        // If you don't handle nested nominal types deeply, then you can get
        // situations where the type changes from underneath you!
        let program = r#"

            struct Point {
                x: Int,
                y: Int,
                z: Int,
            }
            struct MyTuple {
                a: Point,
                b: Point,
            }

            let x: MyTuple = MyTuple { a: Point { x: 1, y: 3, z: 7 }, b: Point { x: 2, y: 5, z: 8}}
            print x

            struct Point {
                x: Int,
                y: Int,
            }
            set x = MyTuple { a: Point { x: 2, y: 5 }, b: Point { x: 3, y: 4 } }
            print x

            ret 0
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 0);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_undefined_struct_1() {
        let program = r#"
            let pt: Point = Point { x: 0, y: 1 }
            print pt
            set pt = Point { x: 3, y: 4 }
            print pt

            ret 0
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 0);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_undefined_struct_2() {
        let program = r#"
            if true {
                struct Point {
                    x: Int
                    y: Int
                } 
            }
            let pt: Point = Point { x: 0, y: 1 }
            print pt
            set pt = Point { x: 3, y: 4 }
            print pt

            ret 0
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 0);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_undefined_struct_3() {
        let program = r#"
            let pt: Point = Point { x: 0, y: 1 }
            print pt
            set pt = Point { x: 3, y: 4 }
            print pt

            struct Point {
                x: Int
                y: Int                
            }

            ret 0
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 0);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_undefined_struct_4() {
        let program = r#"
            fn get_point() -> Point {
                struct Point {
                    x: Int
                    y: Int                
                }
                ret Point { x: 0, y: 0 }
            }
            print get_point()

            ret 0
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 0);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_undefined_struct_5() {
        let program = r#"
            fn get_point(pt: Point) {
                struct Point {
                    x: Int
                    y: Int                
                }
                ret pt
            }
            print get_point()

            ret 0
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 0);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_bad_struct_field_name() {
        let program = r#"
            struct Point {
                x: Int
                y: Int
            } 

            let pt: Point = Point { x: 0, y: 1 }
            print pt
            set pt = Point { x: 3, z: 4 }
            print pt

            ret 0
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 0);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_bad_struct_field_order() {
        let program = r#"
            struct Point {
                x: Int
                y: Int
            } 

            let pt: Point = Point { x: 0, y: 1 }
            print pt
            set pt = Point { y: 4, x: 3 }
            print pt

            ret 0
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 0);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_bad_struct_field_type() {
        let program = r#"
            struct Point {
                x: Int
                y: Int
            } 

            let pt: Point = Point { x: 0, y: 1 }
            print pt
            set pt = Point { x: 3, y: true }
            print pt

            ret 0
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 0);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_bad_struct_field_count_1() {
        let program = r#"
            struct Point {
                x: Int
                y: Int
            } 

            let pt: Point = Point { x: 0, y: 1 }
            print pt
            set pt = Point { x: 3 }
            print pt

            ret 0
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 0);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_bad_struct_field_count_2() {
        let program = r#"
            struct Point {
                x: Int
                y: Int
            } 

            let pt: Point = Point { x: 0, y: 1 }
            print pt
            set pt = Point { x: 3, y: 2, z: 3 }
            print pt

            ret 0
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 0);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_bad_struct_dupe_field_1() {
        let program = r#"
            struct Point {
                x: Int
                y: Int
            } 

            let pt: Point = Point { x: 0, y: 1 }
            print pt
            set pt = Point { x: 3, x: 2, y: 3 }
            print pt

            ret 0
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 0);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_bad_struct_dupe_field_2() {
        let program = r#"
            struct Point {
                x: Int
                y: Int
            } 

            let pt: Point = Point { x: 0, y: 1 }
            print pt
            set pt = Point { x: 3, x: 2 }
            print pt

            ret 0
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 0);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_bad_tuple_length_1() {
        let program = r#"
            let factors: (Int, Bool) = (0, true)
            print factors
            set factors = (2, false, 3)
            print factors

            ret 0
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 0);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_bad_tuple_length_2() {
        let program = r#"
            let factors: (Int, Bool, Int) = (0, true)
            print factors
            set factors = (2, false)
            print factors

            ret 0
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 0);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_bad_tuple_ty() {
        let program = r#"
            let factors: (Int, Bool) = (0, true)
            print factors
            set factors = (2, 2)
            print factors

            ret 0
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 0);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_scoping_1() {
        let program = r#"
            if false {
                let factor: Int = 3
            }
            fn do_thing() -> Int {
                ret factor
            }
            ret do_thing()
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 3);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_scoping_2() {
        let program = r#"
            if true {
                let factor: Int = 3
            }
            fn do_thing() -> Int {
                ret factor
            }
            ret do_thing()
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 3);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_scoping_3() {
        let program = r#"
            if true {
                let factor: Int = 3
            }
            set factor = 4
            ret factor
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 4);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_set_capture() {
        let program = r#"
            let captured: Int = 3
            fn captures_state() {
                set captured = 4
            }
            ret captured
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 4);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_builtin_ty() {
        let program = r#"
            ret mul(true, true)
        "#;

        let (_result, _output) = run_typed(program);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_arg_ty() {
        let program = r#"
            fn square(x: Int) -> Int {
                ret mul(x, x)
            }

            let x: Bool = true
            ret square(x)   
        "#;

        let (_result, _output) = run_typed(program);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_arg_ty_complex() {
        let program = r#"
            fn square(x: Int) -> Int {
                ret mul(x, x)
            }

            let x: Int = 3
            ret square(eq(square(square(square(x)))), 0))   
        "#;

        let (_result, _output) = run_typed(program);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_cond_ty() {
        let program = r#"
            let x: Int = 0
            if x {
                ret -1
            }
            ret 2
        "#;

        let (_result, _output) = run_typed(program);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_let_ty() {
        let program = r#"
            let x: Int = true
            ret 0
        "#;

        let (_result, _output) = run_typed(program);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_set_ty() {
        let program = r#"
            let x: Int = 0
            set x = ()
            ret 0
        "#;

        let (_result, _output) = run_typed(program);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_ret_main_ty() {
        let program = r#"
            let x: Bool = true
            ret x
        "#;

        let (_result, _output) = run_typed(program);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_ret_ty() {
        let program = r#"
            fn f() -> Int {
                ret true
            }
            ret f()
        "#;

        let (_result, _output) = run_typed(program);
    }

    #[test]
    #[should_panic(expected = "Compile Error")]
    fn compile_fail_capture_ty() {
        let program = r#"
            let factor: Bool = true
            fn multi(x: Int) -> Int {
                ret mul(x, factor)
            }

            let x: Int = 7

            ret multi(x)
        "#;

        let (_result, _output) = run_typed(program);
    }

    #[test]
    fn test_basic() {
        // Just tests basic functionality.
        //
        // Whitespace is wonky to make sure the parser is pemissive of that.
        let program = r#"
            fn square(x:Int)->Int{
                ret mul(x, x)
            }

            let x:Int=6
            let cond: Bool = true

            if cond {
                set x = 7
            }

            let y: Int = square(4)
            ret square(x)            
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 49);
    }

    #[test]
    fn test_nested_closure_capture() {
        let program = r#"
            let factor: Int = 3
            fn get_factor() -> Int {
                ret factor
            }
            fn multi(x: Int) -> Int {
                ret mul(x, get_factor())
            }

            let x: Int = 7

            ret multi(x)      
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 21);
    }

    #[test]
    fn test_fn_tys() {
        let program = r#"
            let factor: Int = 3
            fn get_factor() -> Int {
                ret factor
            }
            fn multi(factory: fn() -> Int, x: Int) -> Int {
                ret mul(x, factory())
            }

            let x: Int = 7

            print multi(get_factor, x)  


            fn mega_multi(multiplier: fn(fn() -> Int, Int) -> Int) -> Int {
                fn eleven() -> Int {
                    ret 11
                }
                print multiplier(eleven, 9)
                ret 0
            }

            let _: Int = mega_multi(multi)

            ret 0
        "#;

        let (result, output) = run_typed(program);
        assert_eq!(result, 0);
        assert_eq!(
            output.unwrap(),
            r#"21
99
"#
        )
    }

    #[test]
    fn test_empty_ty() {
        let program = r#"
            let x: () = ()
            set x = ()
            let y: () = x

            fn f1() -> () {
                ret ()
            }
            fn f2(x: ()) -> () {
                ret x
            }
            fn f3(x: (), y: ()) -> () {
                ret y
            }
            fn f4(x: fn() -> ()) -> () {
                ret x()
            }
            fn f5(x: fn(()) -> ()) -> () {
                ret x(())
            }
            fn f6(x: fn(fn() -> ()) -> ()) -> () {
                ret x(f1)
            }
            fn f7(x: fn(fn(()) -> ()) -> ()) -> () {
                ret x(f2)
            }
            fn f8(x: fn(fn((), ()) -> ()) -> (), y: ()) -> () {
                ret x(f3)
            }
            
            let a: () = f1()
            let b: () = f2(())
            let c: () = f3((), ())
            let d: () = f4(f1)
            let e: () = f5(f2)
            let f: () = f6(f4)
            let g: () = f7(f5)
            let h: fn(fn(fn((), ()) -> ()) -> (), ()) -> () = f8

            ret 0           
        "#;

        let (result, _output) = run_typed(program);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_weird_print_ty() {
        // Regression test for that time I accidentally had `print`
        // check that its input was the return type of the parent function.
        let program = r#"
            let factor: Int = 3
            fn get_factor() -> Int {
                ret factor
            }
            fn multi(factory: fn() -> Int, x: Int) -> Int {
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

        let (result, output) = run_typed(program);
        assert_eq!(result, 0);
        assert_eq!(
            output.unwrap(),
            r#"21
99
"#
        )
    }

    #[test]
    fn test_aggregates_basic() {
        let program = r#"
            let factors:(Int,Bool)=( 0, true )
            let factors2: ( Int, Bool ) = (0,true)
            print factors
            set factors = (2, false) 
            print factors

            struct Point {
                x: Int
                y:Str
            }

            let pt: Point = Point { x : 0, y : "hello" }
            let pt2:Point=Point{x:0,y:"hello"}
            print pt
            set pt = Point { x: 3, y: "bye" }
            print pt

            ret 0
        "#;

        let (result, output) = run_typed(program);
        assert_eq!(result, 0);
        assert_eq!(
            output.unwrap(),
            r#"(0, true)
(2, false)
Point { x: 0, y: "hello" }
Point { x: 3, y: "bye" }
"#
        )
    }

    #[test]
    fn test_aggregates_captures() {
        let program = r#"
            fn print_1d_point() -> Int {
                struct Point {
                    x: Int
                }
                let x: Point = Point { x: 1 }
                print x
                ret 0
            }

            let _:Int = print_1d_point()
            let print_point: fn() -> Int = print_1d_point
            let _:Int = print_point()

            if true {
                struct Point {
                    x: Int
                    y: Int
                }

                fn print_2d_point() -> Int {
                    let x: Point = Point { x: 2, y: 4 }
                    print x
                    ret 0
                }

                let _:Int = print_2d_point();
                set print_point = print_2d_point
            }

            struct Point {
                x: Int
                y: Int
                z: Int
            }

            fn print_3d_point() -> Int {
                let x: Point = Point { x: 3, y: 5, z: 7 }
                print x
                ret 0
            }

            let _:Int = print_1d_point()
            let _:Int = print_point()
            let _:Int = print_3d_point()

            ret 0
        "#;

        let (result, output) = run_typed(program);
        assert_eq!(result, 0);
        assert_eq!(
            output.unwrap(),
            r#"Point { x: 1 }
Point { x: 1 }
Point { x: 2, y: 4 }
Point { x: 1 }
Point { x: 2, y: 4 }
Point { x: 3, y: 5, z: 7 }
"#
        )
    }

    #[test]
    fn test_nominal_shadowing() {
        let program = r#"
            struct Point {
                x: Int,
                y: Int,
                z: Int,
            }

            let x1: Point = Point { x: 1, y: 3, z: 7 }
            let x2: Point = Point { x: 2, y: 5, z: 9 }
            print x1
            print x2

            struct Point {
                x: Int,
                y: Int,
            }

            set x2 = x1
            print x1
            print x2

            let y: Point = Point { x: 3, y: 9 }
            print y

            ret 0
        "#;

        let (result, output) = run_typed(program);
        assert_eq!(result, 0);
        assert_eq!(
            output.unwrap(),
            r#"Point { x: 1, y: 3, z: 7 }
Point { x: 2, y: 5, z: 9 }
Point { x: 1, y: 3, z: 7 }
Point { x: 1, y: 3, z: 7 }
Point { x: 3, y: 9 }
"#
        )
    }
}
