//! An absolutely dogshit parser that's easy for me to add things
//! to and not at all the focus of this project.
//!
//! Parsing is line-by-line, so expressions/statements are generally
//! required to be on a single line (unless it's a block statement
//! like a function/if/struct.
//!
//! Syntax is largely based on rust, but with some changes to keep
//! everything super simple.
//!
//! See `fn parse` for the primary entry point.

use crate::*;
use checker::Reg;

use std::collections::BTreeMap;
use std::fmt;

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

/// A span of the input code.
///
/// Values are absolute addresses, which can be converted
/// into an offset by subtracting the address of the input.
#[derive(Debug, Copy, Clone)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

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
pub enum TyName<'p> {
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

#[derive(Debug, Clone)]
pub struct Function<'p> {
    pub name: &'p str,
    pub args: Vec<VarDecl<'p>>,
    pub stmts: Vec<Statement<'p>>,
    pub ty: TyName<'p>,
    pub captures: BTreeMap<&'p str, Reg>,
}

#[derive(Debug, Clone)]
pub struct StructDecl<'p> {
    pub name: &'p str,
    pub fields: Vec<FieldDecl<'p>>,
}

#[derive(Debug, Clone)]
pub struct Statement<'p> {
    pub code: Stmt<'p>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct VarPath<'p> {
    pub ident: &'p str,
    pub fields: Vec<&'p str>,
}

#[derive(Debug, Clone)]
pub enum Stmt<'p> {
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
        is_mut: bool,
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
pub struct Expression<'p> {
    pub code: Expr<'p>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Expr<'p> {
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
pub enum Literal<'p> {
    Int(i64),
    Str(&'p str),
    Bool(bool),
    Empty(()),
}

impl Literal<'_> {
    pub fn ty(&self) -> TyName<'static> {
        match self {
            Literal::Int(_) => TyName::Int,
            Literal::Str(_) => TyName::Str,
            Literal::Bool(_) => TyName::Bool,
            Literal::Empty(_) => TyName::Empty,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VarDecl<'p> {
    pub ident: &'p str,
    pub ty: TyName<'p>,
    pub span: Span,
}
#[derive(Debug, Clone)]
pub struct FieldDecl<'p> {
    pub ident: &'p str,
    pub ty: TyName<'p>,
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
pub struct Builtin {
    pub name: &'static str,
    pub args: &'static [&'static str],
    pub ty: TyName<'static>,
    pub func: for<'e, 'p> fn(args: &[Val<'e, 'p>]) -> Val<'e, 'p>,
}

impl fmt::Debug for Builtin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<builtin>")
    }
}

impl<'p> Program<'p> {
    pub fn parse(&mut self) -> IResult<&'p str, ()> {
        let (i, (Block(stmts), terminal)) = self.parse_block(self.input)?;
        self.main = Some(Function {
            name: "main",
            args: Vec::new(),
            stmts,
            ty: TyName::Func {
                arg_tys: vec![],
                return_ty: Box::new(TyName::Int),
            },
            captures: BTreeMap::new(),
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
                            captures: BTreeMap::new(),
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
            stmt_let_mut,
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

    Ok((
        i,
        Stmt::Let {
            name,
            expr,
            is_mut: false,
        },
    ))
}

fn stmt_let_mut(i: &str) -> IResult<&str, Stmt> {
    let (i, _) = tag("let mut")(i)?;
    let (i, _) = space1(i)?;
    let (i, name) = var_decl(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag("=")(i)?;
    let (i, _) = space0(i)?;
    let (i, expr) = expr(i)?;

    Ok((
        i,
        Stmt::Let {
            name,
            expr,
            is_mut: true,
        },
    ))
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
