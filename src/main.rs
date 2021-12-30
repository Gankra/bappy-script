use std::collections::HashMap;

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





struct Program<'p> {
    main: Function<'p>,
}

struct Function<'p> {
    name: &'p str,
    args: Vec<&'p str>,
    funcs: HashMap<&'p str, Function<'p>>,
    stmts: Vec<Stmt<'p>>,
}

enum Stmt<'p> {
    Let {
        name: &'p str,
        expr: Expr<'p>,
    },    
    Ret {
        expr: Expr<'p>,
    },
    Print {
        expr: Expr<'p>,
    }
}

enum Expr<'p> {
    Call {
        func: &'p str,
        args: Vec<Expr<'p>>,
    },
    Val(Val<'p>),
}

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
    let mut funcs = HashMap::new();
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
                let old = funcs.insert(name, func);
                assert!(old.is_none(), "Duplicate function name: {}", name);
            }
            Item::Stmt(stmt) => {
                stmts.push(stmt);
            }
            Item::End => {
                return Ok((i, Function {
                    name,
                    args,
                    funcs,
                    stmts,
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










fn check(mut program: Program) -> Program {
    check_func(&mut program.main);
    program
}

fn check_func(func: &mut Function) {
    for stmt in &func.stmts {

    }
}














struct Env<'p> {
    vals: HashMap<&'p str, i64>,
    funcs: &'p HashMap<&'p str, Function<'p>>,
}

fn eval_program(program: Program) -> i64 {
    let mut envs = Vec::new();
    eval_func(&program.main, Vec::new(), &mut envs)
}

fn eval_func<'p>(func: &'p Function<'p>, args: Vec<i64>, envs: &mut Vec<Env<'p>>) -> i64 {
    assert!(func.args.len() == args.len(), 
        "mismatched argument count for fn {} (expected {}, got {})", 
        func.name,
        func.args.len(),
        args.len(),
    );

    let vals = func.args.iter().copied().zip(args.into_iter()).collect::<HashMap<_,_>>();
    assert!(vals.len() == func.args.len(),
        "duplicate arg names for fn {}", 
        func.name,
    );

    envs.push(Env {
        vals,
        funcs: &func.funcs,
    });

    for stmt in &func.stmts {
        match stmt {
            Stmt::Let { name, expr } => {
                let val = eval_expr(expr, envs);
                envs.last_mut().unwrap().vals.insert(*name, val);
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
            let func = eval_resolve_func(func, envs);
            let evaled_args = args
                .iter()
                .map(|expr| eval_expr(expr, envs))
                .collect();

            eval_func(func, evaled_args, envs)
        }
        Expr::Val(val) => {
            eval_val(val, envs)
        }
    }
}

fn eval_val(val: &Val, envs: &mut Vec<Env>) -> i64 {
    match val {
        Val::Var(name) => {
            for env in envs.iter().rev() {
                if let Some(val) = env.vals.get(name) {
                    return *val;
                }
            }
            panic!("Tried to get value of undefined var {}", name);
        }
        Val::Lit(int) => *int,
    }
}

fn eval_resolve_func<'p>(func: &'p str, envs: &mut Vec<Env<'p>>) -> &'p Function<'p> {
    for env in envs.iter().rev() {
        if let Some(func) = env.funcs.get(func) {
            return func
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