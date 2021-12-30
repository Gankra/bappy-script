use std::collections::{HashMap, HashSet};

use nom::{
  IResult, Parser, InputTakeAtPosition, AsChar,
  branch::alt,
  bytes::complete::{take_until, tag},
  character::complete::{alphanumeric1, char, digit1, space0, space1},
  combinator::{map, map_res},
  error::ParseError,
  multi::separated_list0,
};

const MAIN_PROGRAM: &str = r#"
    fn x(a, b, c) {
        fn sub(e, f) {
            print c
            ret f
        } 
        ret sub(a, b)
    }

    fn y(z) {
        ret z
    }

    let val1 = 16
    let val2 = x(1, 99, 12)
    let val3 = 21

    let result = x(7, x(val3 , y(x(2, val2, 7)), 8 ), 1)

    let shadowed = 66
    fn captures() {
        ret shadowed
    }
    fn mask(shadowed) {
        ret captures()
    }



    print 777777777
    print mask(33)

    ret result
}

"#;

fn main() {
    run(MAIN_PROGRAM);
}

fn run(input: &str) -> i64 {
    println!("parsing...");
    let (_, bin) = parse(input).unwrap();
    println!("parsed!\n");
    println!("checking...");
    let bin = check(bin);
    println!("checked!\n");
    println!("evaling...");
    let out = eval_program(bin);
    println!("evaled! {}", out);
    out
}




#[derive(Debug, Clone)]
struct Program<'p> {
    main: Function<'p>,
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
    Let {
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
    }
}

#[derive(Debug, Clone)]
enum Expr<'p> {
    Call {
        func: &'p str,
        args: Vec<Expr<'p>>,
    },
    Val(Val<'p>),
}

#[derive(Debug, Clone)]
enum Val<'p> {
    Lit(i64),
    Var(&'p str),
}











// Parse intermediates
enum Item<'p> {
    Func(&'p str, Vec<&'p str>),
    Stmt(Stmt<'p>),
    End,
}

fn parse(i: &str) -> IResult<&str, Program> {
    let (i, main) = parse_func_body(i, "main", Vec::new())?;
    Ok((i, Program { main }))
}

fn parse_func_body<'p>(mut i: &'p str, name: &'p str, args: Vec<&'p str>) 
    -> IResult<&'p str, Function<'p>> {
    let mut stmts = Vec::new();

    loop {
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
                return Ok((i, Function {
                    name,
                    args,
                    stmts,
                    // Captures are populated by the type checker
                    captures: HashSet::new(),
                }));
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
    map(
        alt((stmt_let, stmt_ret, stmt_print)),
        Item::Stmt
    )(i)
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
    alt((expr_call, expr_val))(i)
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

fn expr_val(i: &str) -> IResult<&str, Expr> {
    map(val, Expr::Val)(i)
}

fn val(i: &str) -> IResult<&str, Val> {
    alt((val_lit, val_var))(i)
}

fn val_var(i: &str) -> IResult<&str, Val> {
    map(ident, Val::Var)(i)
}

fn val_lit(i: &str) -> IResult<&str, Val> {
    map(map_res(digit1, |s: &str| s.parse::<i64>()), Val::Lit)(i)
}

fn ident(i: &str) -> IResult<&str, &str> {
    alphanumeric1(i)
}

pub fn padded<F, T, O, E>(
    mut parser: F, 
) -> impl FnMut(T) -> IResult<T, O, E> where
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








struct CheckEnv<'p> {
    vars: HashMap<&'p str, ()>,
}

fn check(mut program: Program) -> Program {
    let mut envs = Vec::new();
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
                // envs.last_mut().unwrap().vars.insert(func.name, ());
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

fn check_expr<'p>(
    expr: &Expr<'p>, 
    envs: &mut Vec<CheckEnv<'p>>, 
    captures: &mut HashSet<&'p str>
){
    match expr {
        Expr::Val(Val::Lit(_)) => {
            // Always valid
        }
        Expr::Val(Val::Var(var_name)) => {
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
        Expr::Call { func, args } => {
            for expr in args {
                check_expr(expr, envs, captures);
            }
        }
    }
}











#[derive(Debug, Clone)]
struct Closure<'p> {
    func: &'p Function<'p>,
    captures: HashMap<&'p str, i64>,
}

#[derive(Debug, Clone)]
struct Env<'p> {
    vals: HashMap<&'p str, i64>,
    funcs: HashMap<&'p str, Closure<'p>>,
}

fn eval_program(program: Program) -> i64 {
    let mut envs = Vec::new();
    eval_func(&program.main, Vec::new(), HashMap::new(), &mut envs)
}

fn eval_func<'p>(
    func: &'p Function<'p>, 
    args: Vec<i64>, 
    captures: HashMap<&'p str, i64>, 
    envs: &mut Vec<Env<'p>>,
) -> i64 {
    assert!(func.args.len() == args.len(), 
        "mismatched argument count for fn {} (expected {}, got {})", 
        func.name,
        func.args.len(),
        args.len(),
    );

    let mut vals = func.args.iter().copied().zip(args.into_iter()).collect::<HashMap<_,_>>();
    assert!(vals.len() == func.args.len(),
        "duplicate arg names for fn {}", 
        func.name,
    );

    vals.extend(captures.into_iter());

    envs.push(Env {
        vals,
        funcs: HashMap::new(),
    });

    for stmt in &func.stmts {
        match stmt {
            Stmt::Let { name, expr } => {
                let val = eval_expr(expr, envs);
                envs.last_mut().unwrap().vals.insert(*name, val);
            }
            Stmt::Func { func } => {
                let captures = func.captures
                    .iter()
                    .map(|&var| (var, eval_var(var, envs)))
                    .collect();

                envs.last_mut().unwrap().funcs.insert(func.name, Closure {
                    captures,
                    func,
                });
            }
            Stmt::Print { expr } => {
                let val = eval_expr(expr, envs);
                println!("{}", val);
            }
            Stmt::Ret { expr } => {
                let val = eval_expr(expr, envs);
                envs.pop();
                return val;
            }
        }
    }

    panic!("Ran out of statements to evaulate for function {}", func.name);
}

fn eval_expr<'p>(expr: &'p Expr<'p>, envs: &mut Vec<Env<'p>>) -> i64 {
    match expr {
        Expr::Call { func, args } => {
            let closure = eval_resolve_func(func, envs);
            let evaled_args = args
                .iter()
                .map(|expr| eval_expr(expr, envs))
                .collect();

            eval_func(closure.func, evaled_args, closure.captures, envs)
        }
        Expr::Val(val) => {
            eval_val(val, envs)
        }
    }
}

fn eval_val(val: &Val, envs: &mut Vec<Env>) -> i64 {
    match val {
        Val::Var(name) => eval_var(name, envs),
        Val::Lit(int) => *int,
    }
}

fn eval_var(var: &str, envs: &mut Vec<Env>) -> i64 {
    for env in envs.iter().rev() {
        if let Some(val) = env.vals.get(var) {
            return *val;
        }
    }
    panic!("Tried to get value of undefined var {}", var);
}

fn eval_resolve_func<'a, 'p>(func: &'p str, envs: &mut Vec<Env<'p>>) -> Closure<'p> {
    for env in envs.iter().rev() {
        if let Some(func) = env.funcs.get(func) {
            return func.clone()
        }
    }
    panic!("Could not resolve function name: {}", func);
}









#[cfg(test)]
mod test {
    use super::run;

    // TODO: this fails, it shouldn't
    #[test]
    #[should_panic]
    fn test_first_class_basic() {
        let program = r#"
            let capture = 123
            fn do_it() {
                ret capture
            }
            fn higher(func) {
                func()
            }

            ret higher(do_it)
        }
        "#;

        let result = run(program);
        assert_eq!(result, 123);
    }

    // TODO: this fails, it shouldn't
    #[test]
    #[should_panic]
    fn test_first_class_with_args() {
        let program = r#"
            let capture = 123;
            fn do_it(x, y) {
                ret x
            }
            fn higher(func) {
                func(777, 999)
            }

            ret higher(do_it)
        }
        "#;

        let result = run(program);
        assert_eq!(result, 777);
    }

    // TODO: this fails, it shouldn't
    #[test]
    #[should_panic]
    fn test_first_class_with_captures() {
        let program = r#"
            let capture = 666
            fn captured_func() {
                ret capture
            }
            fn do_it(x, y) {
                ret captured_func()
            }
            fn higher(func) {
                func(777, 999)
            }

            ret higher(do_it)
        }
        "#;

        let result = run(program);
        assert_eq!(result, 666);
    }
       
        


    // TODO: this fails, it shouldn't
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
        }
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
        }
        "#;

        let result = run(program);
        assert_eq!(result, 99);
    }


}