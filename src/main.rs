use std::collections::{HashMap, HashSet};
use std::fmt::{self, Write};

use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{alpha1, alphanumeric1, char, space0, space1},
    combinator::{map, recognize, rest},
    error::ParseError,
    multi::{many0, separated_list0},
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
    fn square(x: Int) -> Int {
        ret mul(x, x)
    }

    let x: Int = 7

    ret square(x)
"#;

fn main() {
    run_typed(MAIN_PROGRAM);
}

#[allow(dead_code)]
fn run(input: &str) -> (i64, Option<String>) {
    println!("parsing...");
    let (_, mut bin) = parse(input).expect("Parse Error");
    println!("parsed!\n");

    bin.builtins = builtins();
    bin.output = Some(String::new());
    bin.typed = false;

    println!("checking...");
    let mut bin = bin.check();
    println!("checked!\n");

    println!("evaling...");
    let out = bin.eval();
    println!("evaled!");
    println!("{}", out);

    (out, bin.output)
}

#[allow(dead_code)]
fn run_typed(input: &str) -> (i64, Option<String>) {
    println!("parsing...");
    let (_, mut bin) = parse(input).expect("Parse Error");
    println!("parsed!\n");

    bin.builtins = builtins();
    bin.output = Some(String::new());
    bin.typed = true;

    println!("checking...");
    let mut bin = bin.check();
    println!("checked!\n");

    println!("evaling...");
    let out = bin.eval();
    println!("evaled!");
    println!("{}", out);

    (out, bin.output)
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

#[derive(Debug, Clone)]
struct Program<'p> {
    main: Option<Function<'p>>,
    typed: bool,
    builtins: Vec<Builtin>,
    output: Option<String>,
}

#[derive(Debug, Clone)]
struct Function<'p> {
    name: &'p str,
    args: Vec<VarDecl<'p>>,
    stmts: Vec<Stmt<'p>>,
    ty: Ty,
    captures: HashSet<&'p str>,
}

#[derive(Debug, Clone)]
enum Stmt<'p> {
    If {
        expr: Expr<'p>,
        stmts: Vec<Stmt<'p>>,
        else_stmts: Vec<Stmt<'p>>,
    },
    Loop {
        stmts: Vec<Stmt<'p>>,
    },
    Let {
        name: VarDecl<'p>,
        expr: Expr<'p>,
    },
    Set {
        name: &'p str,
        expr: Expr<'p>,
    },
    Func {
        func: Function<'p>,
    },
    Ret {
        expr: Expr<'p>,
    },
    Print {
        expr: Expr<'p>,
    },
    Break,
    Continue,
}

#[derive(Debug, Clone)]
enum Expr<'p> {
    Call { func: &'p str, args: Vec<Expr<'p>> },
    Lit(Literal<'p>),
    Var(&'p str),
}

#[derive(Debug, Clone)]
enum Literal<'p> {
    Int(i64),
    Str(&'p str),
    Bool(bool),
    Empty(()),
}

impl Literal<'_> {
    fn ty(&self) -> Ty {
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
    ty: Ty,
}

// Parse intermediates
enum Item<'p> {
    Comment(&'p str),
    Func(&'p str, Vec<VarDecl<'p>>, Ty),
    Stmt(Stmt<'p>),
    If(Expr<'p>),
    Loop,
    Else,
    End,
}

struct Block<'p>(Vec<Stmt<'p>>);

#[derive(Clone)]
struct Builtin {
    name: &'static str,
    args: &'static [&'static str],
    ty: Ty,
    func: for<'e, 'p> fn(args: &[Val<'e, 'p>]) -> Val<'e, 'p>,
}

impl fmt::Debug for Builtin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<builtin>")
    }
}

fn parse(i: &str) -> IResult<&str, Program> {
    let (i, (Block(stmts), terminal)) = parse_block(i)?;
    let main = Some(Function {
        name: "main",
        args: Vec::new(),
        stmts,
        ty: Ty::Func {
            arg_tys: vec![],
            return_ty: Box::new(Ty::Int),
        },
        captures: HashSet::new(),
    });

    assert!(
        matches!(terminal, Item::End),
        "Parse Error: function ending eith an `else`"
    );

    Ok((
        i,
        Program {
            main,
            // other state populated by caller
            typed: false,
            builtins: Vec::new(),
            output: None,
        },
    ))
}

fn parse_block<'p>(mut i: &'p str) -> IResult<&'p str, (Block<'p>, Item<'p>)> {
    let mut stmts = Vec::new();

    loop {
        if i.trim().is_empty() {
            return Ok((i, (Block(stmts), Item::End)));
        }

        let (new_i, line) = take_until("\n")(i)?;
        println!("{}", line);
        i = &new_i[1..];
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        match item(line)?.1 {
            Item::Func(name, args, return_ty) => {
                let (new_i, (Block(block_stmts), terminal)) = parse_block(i)?;
                i = new_i;

                assert!(
                    matches!(terminal, Item::End),
                    "Parse Error: function ending eith an `else`"
                );

                stmts.push(Stmt::Func {
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
                });
            }
            Item::If(expr) => {
                let (new_i, (Block(block_stmts), terminal)) = parse_block(i)?;
                i = new_i;

                let else_stmts = if let Item::Else = terminal {
                    let (new_i, (Block(else_stmts), terminal)) = parse_block(i)?;
                    i = new_i;

                    assert!(
                        matches!(terminal, Item::End),
                        "Parse Error: `else` ending eith an `else`"
                    );
                    else_stmts
                } else {
                    Vec::new()
                };

                stmts.push(Stmt::If {
                    expr,
                    stmts: block_stmts,
                    else_stmts,
                })
            }
            Item::Loop => {
                let (new_i, (Block(block_stmts), terminal)) = parse_block(i)?;
                i = new_i;

                assert!(
                    matches!(terminal, Item::End),
                    "Parse Error: loop ending eith an `else`"
                );

                stmts.push(Stmt::Loop { stmts: block_stmts });
            }
            Item::Stmt(stmt) => {
                stmts.push(stmt);
            }
            Item::Comment(_comment) => {
                // discard it
            }
            item @ Item::End | item @ Item::Else => return Ok((i, (Block(stmts), item))),
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

fn return_ty(i: &str) -> IResult<&str, Ty> {
    let (i, _) = space0(i)?;
    let (i, _) = tag("->")(i)?;
    let (i, _) = space0(i)?;
    ty_decl(i)
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

fn expr(i: &str) -> IResult<&str, Expr> {
    alt((expr_call, expr_lit, expr_var))(i)
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
    let (i, id) = recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ))(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag(":")(i)?;
    let (i, _) = space0(i)?;
    let (i, ty) = ty_decl(i)?;

    Ok((i, VarDecl { ident: id, ty }))
}

fn ty_decl(i: &str) -> IResult<&str, Ty> {
    alt((
        ty_decl_int,
        ty_decl_str,
        ty_decl_empty,
        ty_decl_bool,
        ty_decl_func,
    ))(i)
}

fn ty_decl_int(i: &str) -> IResult<&str, Ty> {
    map(tag("Int"), |_| Ty::Int)(i)
}
fn ty_decl_str(i: &str) -> IResult<&str, Ty> {
    map(tag("Str"), |_| Ty::Str)(i)
}
fn ty_decl_bool(i: &str) -> IResult<&str, Ty> {
    map(tag("Bool"), |_| Ty::Bool)(i)
}
fn ty_decl_empty(i: &str) -> IResult<&str, Ty> {
    map(tag("()"), |_| Ty::Empty)(i)
}
fn ty_decl_func(i: &str) -> IResult<&str, Ty> {
    // TODO
    map(tag("fn"), |_| Ty::Func {
        arg_tys: Vec::new(),
        return_ty: Box::new(Ty::Unknown),
    })(i)
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

#[derive(Clone, Debug, PartialEq, Eq)]
enum Ty {
    Int,
    Str,
    Bool,
    Empty,
    Func {
        arg_tys: Vec<Ty>,
        return_ty: Box<Ty>,
    },
    Unknown,
}

struct CheckEnv<'p> {
    vars: HashMap<&'p str, Ty>,
}

impl<'p> Program<'p> {
    fn check(mut self) -> Self {
        let builtins = self
            .builtins
            .iter()
            .map(|builtin| (builtin.name, builtin.ty.clone()))
            .collect();
        let mut envs = vec![CheckEnv { vars: builtins }];
        let mut main = self.main.take().unwrap();
        self.check_func(&mut main, &mut envs);
        self.main = Some(main);
        self
    }

    fn check_func(&mut self, func: &mut Function<'p>, envs: &mut Vec<CheckEnv<'p>>) {
        let vars = func
            .args
            .iter()
            .map(|decl| (decl.ident, decl.ty.clone()))
            .collect();
        envs.push(CheckEnv { vars });
        let mut captures = HashSet::new();

        let return_ty = if let Ty::Func { return_ty, .. } = &func.ty {
            &*return_ty
        } else {
            &Ty::Unknown
        };

        self.check_block(&mut func.stmts, envs, &mut captures, return_ty);

        func.captures = captures;
        envs.pop();
    }

    fn check_block(
        &mut self,
        stmts: &mut [Stmt<'p>],
        envs: &mut Vec<CheckEnv<'p>>,
        captures: &mut HashSet<&'p str>,
        return_ty: &Ty,
    ) {
        for stmt in stmts {
            match stmt {
                Stmt::If {
                    expr,
                    stmts,
                    else_stmts,
                } => {
                    let expr_ty = self.check_expr(expr, envs, captures);

                    if self.typed {
                        assert!(
                            &expr_ty == &Ty::Bool,
                            "Compile Error: If type mismatch (expected {:?}, got {:?})",
                            Ty::Bool,
                            expr_ty
                        );
                    }

                    self.check_block(stmts, envs, captures, return_ty);
                    self.check_block(else_stmts, envs, captures, return_ty);
                }
                Stmt::Loop { stmts } => {
                    self.check_block(stmts, envs, captures, return_ty);
                }
                Stmt::Func { func } => {
                    // We push a func's name after checking it to avoid
                    // infinite capture recursion. This means naive recursion
                    // is illegal.
                    self.check_func(func, envs);
                    envs.last_mut()
                        .unwrap()
                        .vars
                        .insert(func.name, func.ty.clone());
                }
                Stmt::Let { name, expr } => {
                    let expr_ty = self.check_expr(expr, envs, captures);
                    let decl_ty = &name.ty;

                    if self.typed {
                        assert!(
                            &expr_ty == decl_ty,
                            "Compile Error: Let type mismatch (expected {:?}, got {:?})",
                            decl_ty,
                            expr_ty
                        );
                    }

                    envs.last_mut().unwrap().vars.insert(name.ident, expr_ty);
                }
                Stmt::Set { name, expr } => {
                    let expr_ty = self.check_expr(expr, envs, captures);

                    if let Some(old_ty) = envs.last().unwrap().vars.get(name) {
                        if self.typed {
                            assert!(
                                &expr_ty == old_ty,
                                "Compile Error: Set type mismatch (expected {:?}, got {:?})",
                                old_ty,
                                expr_ty
                            );
                        }

                        envs.last_mut().unwrap().vars.insert(name, expr_ty);
                    } else {
                        panic!(
                            "Compile Error: trying to set an undefined local variable {}",
                            name
                        );
                    }
                }
                Stmt::Ret { expr } | Stmt::Print { expr } => {
                    let expr_ty = self.check_expr(expr, envs, captures);

                    if self.typed {
                        assert!(
                            &expr_ty == return_ty,
                            "Compile Error: Return type mismatch (expected {:?}, got {:?})",
                            return_ty,
                            expr_ty
                        );
                    }
                }
                Stmt::Break | Stmt::Continue => {
                    // Nothing to analyze
                }
            }
        }
    }

    fn check_expr(
        &mut self,
        expr: &Expr<'p>,
        envs: &mut Vec<CheckEnv<'p>>,
        captures: &mut HashSet<&'p str>,
    ) -> Ty {
        match expr {
            Expr::Lit(lit) => {
                return lit.ty();
            }
            Expr::Var(var_name) => {
                for (depth, env) in envs.iter().rev().enumerate() {
                    if let Some(ty) = env.vars.get(var_name) {
                        if depth == 0 {
                            // Do nothing, not a capture
                        } else {
                            captures.insert(var_name);
                        }
                        return ty.clone();
                    }
                }
                panic!("Compile Error: Use of undefined variable {}", var_name);
            }
            Expr::Call { func, args } => {
                for (depth, env) in envs.iter().rev().enumerate() {
                    if let Some(func_ty) = env.vars.get(func) {
                        let (arg_tys, return_ty) = if let Ty::Func { arg_tys, return_ty } = func_ty
                        {
                            (arg_tys.clone(), (**return_ty).clone())
                        } else if self.typed {
                            panic!("Compile Error: Function call must have Func type!");
                        } else {
                            (Vec::new(), Ty::Unknown)
                        };

                        if depth == 0 {
                            // Do nothing, not a capture
                        } else {
                            captures.insert(func);
                        }

                        for (idx, expr) in args.iter().enumerate() {
                            let expr_ty = self.check_expr(expr, envs, captures);
                            if self.typed {
                                let arg_ty = &arg_tys[idx];
                                assert!(&expr_ty == arg_ty, "Compile Error: Argument type mismatch (expected {:?}, got {:?})", arg_ty, expr_ty);
                            }
                        }
                        return return_ty;
                    }
                }
                panic!("Compile Error: Call of undefined function {}", func);
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

        if let Val::Int(int) = out {
            self.main = Some(main);
            int
        } else {
            panic!("main must evaluate to an int!");
        }
    }

    fn eval_func<'e>(
        &mut self,
        func: &'e Function<'p>,
        args: Vec<Val<'e, 'p>>,
        captures: HashMap<&'p str, Val<'e, 'p>>,
        envs: &mut Vec<Env<'e, 'p>>,
    ) -> Val<'e, 'p> {
        assert!(
            func.args.len() == args.len(),
            "Runtime Error: mismatched argument count for fn {} (expected {}, got {})",
            func.name,
            func.args.len(),
            args.len(),
        );

        let mut vals = func
            .args
            .iter()
            .map(|arg| arg.ident)
            .zip(args.into_iter())
            .collect::<HashMap<_, _>>();
        assert!(
            vals.len() == func.args.len(),
            "Runtime Error: duplicate arg names for fn {}",
            func.name,
        );

        vals.extend(captures.into_iter());

        envs.push(Env { vals });
        let result = self.eval_block(&func.stmts, envs);
        envs.pop();

        match result {
            ControlFlow::Return(val) => val,
            ControlFlow::Break => {
                panic!("Runtime Error: break used outside of a loop");
            }
            ControlFlow::Continue => {
                panic!("Runtime Error: continue used outside of a loop");
            }
            ControlFlow::None => {
                panic!(
                    "Runtime Error: function didn't return a value: {}",
                    func.name
                );
            }
        }
    }

    fn eval_block<'e>(
        &mut self,
        stmts: &'e [Stmt<'p>],
        envs: &mut Vec<Env<'e, 'p>>,
    ) -> ControlFlow<'e, 'p> {
        for stmt in stmts {
            match stmt {
                Stmt::Let { name, expr } => {
                    let val = self.eval_expr(expr, envs);
                    envs.last_mut().unwrap().vals.insert(name.ident, val);
                }
                Stmt::Set { name, expr } => {
                    let val = self.eval_expr(expr, envs);
                    let old = envs.last_mut().unwrap().vals.insert(*name, val);
                    assert!(
                        old.is_some(),
                        "Runtime Error: Tried to set an undefined local variable {}",
                        name
                    );
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
                            panic!(
                                "Runtime Error: Tried to branch on non-boolean {}",
                                self.format_val(&val, true, 0)
                            );
                        }
                    };

                    match result {
                        ControlFlow::None => { /* do nothing */ }
                        // All other control flow ends the block immediately
                        flow => return flow,
                    }
                }
                Stmt::Loop { stmts } => {
                    loop {
                        let result = self.eval_block(stmts, envs);
                        match result {
                            ControlFlow::Return(val) => return ControlFlow::Return(val),
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
                    return ControlFlow::Return(val);
                }
                Stmt::Break => {
                    return ControlFlow::Break;
                }
                Stmt::Continue => {
                    return ControlFlow::Continue;
                }
            }
        }

        // Nothing special happened, continue execution
        ControlFlow::None
    }

    fn eval_expr<'e>(&mut self, expr: &Expr<'p>, envs: &mut Vec<Env<'e, 'p>>) -> Val<'e, 'p> {
        match expr {
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
                        panic!(
                            "Runtime Error: Tried to call a non-function {}: {}",
                            func_name,
                            self.format_val(&func, true, 0)
                        );
                    }
                }
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
        panic!("Runtime Error: Use of undefined var {}", var);
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
    use super::run;

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
    use super::run_typed;

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
    fn test_basic() {
        // Just tests basic functionality.
        //
        // Whitespace is wonky to make sure the parser is pemissive of that.
        let program = r#"
            fn square(x: Int) -> Int {
                ret mul(x, x)
            }

            let x: Int = 6
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
}
