use std::collections::{HashMap, HashSet};
use std::fmt;

use nom::{
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{alphanumeric1, char, digit1, space0, space1},
    combinator::{map, map_res},
    error::ParseError,
    multi::separated_list0,
    AsChar, IResult, InputTakeAtPosition, Parser,
};

const MAIN_PROGRAM: &str = r#"
    fn true(if, else) {
        ret if()
    }
    fn false(if, else) {
        ret else()
    }

    let condition = true
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

fn main() {
    run(MAIN_PROGRAM);
}

fn builtin_add<'p>(args: &[Val<'p>]) -> Val<'p> {
    assert!(args.len() == 2, "Builtin [add]: wrong number of args");
    if let (Val::Int(lhs), Val::Int(rhs)) = (&args[0], &args[1]) {
        Val::Int(lhs + rhs)
    } else {
        panic!("Builtin [add]: wrong type of args")
    }
}

fn builtin_mul<'p>(args: &[Val<'p>]) -> Val<'p> {
    assert!(args.len() == 2, "Builtin [mul]: wrong number of args");
    if let (Val::Int(lhs), Val::Int(rhs)) = (&args[0], &args[1]) {
        Val::Int(lhs * rhs)
    } else {
        panic!("Builtin [mul]: wrong type of args")
    }
}

fn builtin_sub<'p>(args: &[Val<'p>]) -> Val<'p> {
    assert!(args.len() == 2, "Builtin [sub]: wrong number of args");
    if let (Val::Int(lhs), Val::Int(rhs)) = (&args[0], &args[1]) {
        Val::Int(lhs - rhs)
    } else {
        panic!("Builtin [sub]: wrong type of args")
    }
}

fn builtins() -> &'static [(&'static str, Builtin)] {
    &[
        (
            "add",
            Builtin {
                args: &["lhs", "rhs"],
                func: builtin_add,
            },
        ),
        (
            "sub",
            Builtin {
                args: &["lhs", "rhs"],
                func: builtin_sub,
            },
        ),
        (
            "mul",
            Builtin {
                args: &["lhs", "rhs"],
                func: builtin_mul,
            },
        ),
    ]
}

fn run(input: &str) -> i64 {
    println!("parsing...");
    let (_, mut bin) = parse(input).unwrap();
    println!("parsed!\n");

    bin.builtins = builtins();

    println!("checking...");
    let bin = check(bin);
    println!("checked!\n");

    println!("evaling...");
    let out = eval_program(&bin);
    println!("evaled!");

    print_val(&out);

    if let Val::Int(int) = out {
        int
    } else {
        panic!("main must evaluate to an int!");
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

#[derive(Debug, Clone)]
struct Program<'p> {
    main: Function<'p>,
    builtins: &'static [(&'static str, Builtin)],
}

#[derive(Debug, Clone)]
struct Function<'p> {
    name: &'p str,
    args: Vec<&'p str>,
    stmts: Vec<Stmt<'p>>,
    captures: HashSet<&'p str>,
}

#[derive(Debug, Clone)]
enum Stmt<'p> {
    Let { name: &'p str, expr: Expr<'p> },
    Func { func: Function<'p> },
    Ret { expr: Expr<'p> },
    Print { expr: Expr<'p> },
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
}

// Parse intermediates
enum Item<'p> {
    Func(&'p str, Vec<&'p str>),
    Stmt(Stmt<'p>),
    End,
}

#[derive(Clone)]
struct Builtin {
    args: &'static [&'static str],
    func: for<'p> fn(args: &[Val<'p>]) -> Val<'p>,
}

impl fmt::Debug for Builtin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<builtin>")
    }
}

fn parse(i: &str) -> IResult<&str, Program> {
    let (i, main) = parse_func_body(i, "main", Vec::new())?;
    Ok((
        i,
        Program {
            main,
            builtins: &[],
        },
    ))
}

fn parse_func_body<'p>(
    mut i: &'p str,
    name: &'p str,
    args: Vec<&'p str>,
) -> IResult<&'p str, Function<'p>> {
    let mut stmts = Vec::new();

    loop {
        if i.trim().is_empty() {
            return Ok((
                i,
                Function {
                    name,
                    args,
                    stmts,
                    // Captures are populated by the type checker
                    captures: HashSet::new(),
                },
            ));
        }

        let (new_i, line) = take_until("\n")(i)?;
        println!("{}", line);
        i = &new_i[1..];
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        match item(line)?.1 {
            Item::Func(name, args) => {
                let (new_i, func) = parse_func_body(i, name, args)?;
                i = new_i;
                stmts.push(Stmt::Func { func });
            }
            Item::Stmt(stmt) => {
                stmts.push(stmt);
            }
            Item::End => {
                return Ok((
                    i,
                    Function {
                        name,
                        args,
                        stmts,
                        // Captures are populated by the type checker
                        captures: HashSet::new(),
                    },
                ));
            }
        }
    }
}

fn item(i: &str) -> IResult<&str, Item> {
    alt((func, stmt, end))(i)
}

fn func(i: &str) -> IResult<&str, Item> {
    let (i, _) = tag("fn")(i)?;
    let (i, _) = space1(i)?;
    let (i, name) = ident(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag("(")(i)?;
    let (i, _) = space0(i)?;
    let (i, args) = separated_list0(char(','), padded(ident))(i)?;
    let (i, _) = tag(")")(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag("{")(i)?;

    Ok((i, Item::Func(name, args)))
}

fn stmt(i: &str) -> IResult<&str, Item> {
    map(alt((stmt_let, stmt_ret, stmt_print)), Item::Stmt)(i)
}

fn end(i: &str) -> IResult<&str, Item> {
    let (i, _) = tag("}")(i)?;
    Ok((i, Item::End))
}

fn stmt_let(i: &str) -> IResult<&str, Stmt> {
    let (i, _) = tag("let")(i)?;
    let (i, _) = space1(i)?;
    let (i, name) = ident(i)?;
    let (i, _) = space0(i)?;
    let (i, _) = tag("=")(i)?;
    let (i, _) = space0(i)?;
    let (i, expr) = expr(i)?;

    Ok((i, Stmt::Let { name, expr }))
}

fn stmt_ret(i: &str) -> IResult<&str, Stmt> {
    let (i, _) = tag("ret")(i)?;
    let (i, _) = space1(i)?;
    let (i, expr) = expr(i)?;

    Ok((i, Stmt::Ret { expr }))
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
    map(alt((expr_lit_int, expr_lit_str)), Expr::Lit)(i)
}

fn expr_lit_int(i: &str) -> IResult<&str, Literal> {
    map(map_res(digit1, |s: &str| s.parse::<i64>()), Literal::Int)(i)
}

fn expr_lit_str(i: &str) -> IResult<&str, Literal> {
    let (i, _) = tag("\"")(i)?;
    let (i, string) = take_until("\"")(i)?;
    let (i, _) = tag("\"")(i)?;
    Ok((i, Literal::Str(string)))
}

fn ident(i: &str) -> IResult<&str, &str> {
    alphanumeric1(i)
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

struct CheckEnv<'p> {
    vars: HashMap<&'p str, ()>,
}

fn check(mut program: Program) -> Program {
    let builtins = program
        .builtins
        .iter()
        .map(|(name, _)| (*name, ()))
        .collect();
    let mut envs = vec![CheckEnv { vars: builtins }];
    check_func(&mut program.main, &mut envs);
    program
}

fn check_func<'p>(func: &mut Function<'p>, envs: &mut Vec<CheckEnv<'p>>) {
    let vars = func.args.iter().map(|&name| (name, ())).collect();
    envs.push(CheckEnv { vars });
    let mut captures = HashSet::new();

    for stmt in &mut func.stmts {
        match stmt {
            Stmt::Let { name, expr } => {
                check_expr(expr, envs, &mut captures);
                envs.last_mut().unwrap().vars.insert(name, ());
            }
            Stmt::Func { func } => {
                envs.last_mut().unwrap().vars.insert(func.name, ());
                check_func(func, envs);
            }
            Stmt::Ret { expr } | Stmt::Print { expr } => {
                check_expr(expr, envs, &mut captures);
            }
        }
    }

    func.captures = captures;
    envs.pop();
}

fn check_expr<'p>(expr: &Expr<'p>, envs: &mut Vec<CheckEnv<'p>>, captures: &mut HashSet<&'p str>) {
    match expr {
        Expr::Lit(..) => {
            // Always valid
        }
        Expr::Var(var_name) => {
            for (depth, env) in envs.iter().rev().enumerate() {
                if env.vars.get(var_name).is_some() {
                    if depth == 0 {
                        // Do nothing, not a capture
                    } else {
                        captures.insert(var_name);
                    }
                    return;
                }
            }
            panic!("Compile Error: Use of undefined variable {}", var_name);
        }
        Expr::Call { args, .. } => {
            for expr in args {
                check_expr(expr, envs, captures);
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
enum Val<'p> {
    Int(i64),
    Str(&'p str),
    Func(Closure<'p>),
    Builtin(Builtin),
}

#[derive(Debug, Clone)]
struct Closure<'p> {
    func: &'p Function<'p>,
    captures: HashMap<&'p str, Val<'p>>,
}

#[derive(Debug, Clone)]
struct Env<'p> {
    vals: HashMap<&'p str, Val<'p>>,
}

fn eval_program<'p>(program: &'p Program<'p>) -> Val<'p> {
    let builtins = program
        .builtins
        .iter()
        .map(|(name, builtin)| (*name, Val::Builtin(builtin.clone())))
        .collect();
    let mut envs = vec![Env { vals: builtins }];
    eval_func(&program.main, Vec::new(), HashMap::new(), &mut envs)
}

fn eval_func<'p>(
    func: &'p Function<'p>,
    args: Vec<Val<'p>>,
    captures: HashMap<&'p str, Val<'p>>,
    envs: &mut Vec<Env<'p>>,
) -> Val<'p> {
    assert!(
        func.args.len() == args.len(),
        "mismatched argument count for fn {} (expected {}, got {})",
        func.name,
        func.args.len(),
        args.len(),
    );

    let mut vals = func
        .args
        .iter()
        .copied()
        .zip(args.into_iter())
        .collect::<HashMap<_, _>>();
    assert!(
        vals.len() == func.args.len(),
        "duplicate arg names for fn {}",
        func.name,
    );

    vals.extend(captures.into_iter());

    envs.push(Env { vals });

    for stmt in &func.stmts {
        match stmt {
            Stmt::Let { name, expr } => {
                let val = eval_expr(expr, envs);
                envs.last_mut().unwrap().vals.insert(*name, val);
            }
            Stmt::Func { func } => {
                let captures = func
                    .captures
                    .iter()
                    .map(|&var| (var, eval_resolve_var(var, envs)))
                    .collect();

                envs.last_mut()
                    .unwrap()
                    .vals
                    .insert(func.name, Val::Func(Closure { captures, func }));
            }
            Stmt::Print { expr } => {
                let val = eval_expr(expr, envs);
                print_val(&val);
            }
            Stmt::Ret { expr } => {
                let val = eval_expr(expr, envs);
                envs.pop();
                return val;
            }
        }
    }

    panic!(
        "Ran out of statements to evaulate for function {}",
        func.name
    );
}

fn eval_expr<'p>(expr: &'p Expr<'p>, envs: &mut Vec<Env<'p>>) -> Val<'p> {
    match expr {
        Expr::Call {
            func: func_name,
            args,
        } => {
            let func = eval_resolve_var(func_name, envs);
            let evaled_args = args.iter().map(|expr| eval_expr(expr, envs)).collect();

            match func {
                Val::Func(closure) => eval_func(closure.func, evaled_args, closure.captures, envs),
                Val::Builtin(builtin) => (builtin.func)(&evaled_args),
                _ => {
                    panic!("Tried to call a non-function: {}", func_name);
                }
            }
        }
        Expr::Var(var) => eval_resolve_var(var, envs),
        Expr::Lit(lit) => match lit {
            Literal::Int(int) => Val::Int(*int),
            Literal::Str(string) => Val::Str(*string),
        },
    }
}

fn eval_resolve_var<'p>(var: &'p str, envs: &mut Vec<Env<'p>>) -> Val<'p> {
    for env in envs.iter().rev() {
        if let Some(val) = env.vals.get(var) {
            return val.clone();
        }
    }
    panic!("Use of undefined var: {}", var);
}

fn print_val(val: &Val) {
    match val {
        Val::Int(int) => {
            println!("{}", int);
        }
        Val::Str(string) => {
            println!("{}", string);
        }
        Val::Func(func) => {
            println!("fn {}", func.func.name);
        }
        Val::Builtin(..) => {
            println!("<builtin>");
        }
    }
}

#[cfg(test)]
mod test {
    use super::run;

    #[test]
    fn test_builtin_math() {
        let program = r#"
            ret sub(mul(add(4, 7), 13), 9)
        "#;

        let result = run(program);
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

        let result = run(program);
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

        let result = run(program);
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

        let result = run(program);
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

        let result = run(program);
        assert_eq!(result, 66);
    }

    // TODO: this fails, it shouldn't
    #[test]
    #[should_panic]
    fn test_negative() {
        let program = r#"
            ret -1
        }
        "#;

        let result = run(program);
        assert_eq!(result, -1);
    }

    #[test]
    fn test_fake_bools() {
        let program = r#"
            fn true(if, else) {
                ret if()
            }
            fn false(if, else) {
                ret else()
            }

            let condition = true
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

        let result = run(program);
        assert_eq!(result, 70);
    }

    #[test]
    fn test_str() {
        let program = r#"
            print "hello"
            ret 1
        "#;

        let result = run(program);
        assert_eq!(result, 1);
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

        let result = run(program);
        assert_eq!(result, 99);
    }
}
