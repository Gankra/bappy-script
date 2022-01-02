//! A simple untyped interpretter of the """AST"""
//!
//! The interpretter paranoidly assumes the "checker" did a bad job and will
//! double check things and report runtime errors for things it should have caught.
//!
//! `fn eval` is the main entry point.

use crate::*;

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;

/// A runtime value
#[derive(Debug, Clone)]
pub enum Val<'e, 'p> {
    Int(i64),
    Str(&'p str),
    Bool(bool),
    Empty(()),
    Tuple(Vec<Val<'e, 'p>>),
    Struct(&'p str, BTreeMap<&'p str, Val<'e, 'p>>),
    Func(Closure<'e, 'p>),
    Builtin(Builtin),
}

/// A runtime closure
#[derive(Debug, Clone)]
pub struct Closure<'e, 'p> {
    pub func: &'e Function<'p>,
    pub captures: HashMap<&'p str, Val<'e, 'p>>,
}

/// A scope in the program's execution
#[derive(Debug, Clone)]
pub struct Env<'e, 'p> {
    vals: HashMap<&'p str, Val<'e, 'p>>,
}

/// What control flow caused us to exit a block, allowing us to bubble up
/// the control flow to the relevant part of the program (loop, func).
pub enum ControlFlow<'e, 'p> {
    Return(Val<'e, 'p>),
    Break,
    Continue,
    None,
}

impl<'p> Program<'p> {
    /// Run the program!
    pub fn eval(&mut self) -> i64 {
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

pub fn builtins() -> Vec<Builtin> {
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
