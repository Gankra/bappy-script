fn run(input: &str) -> (i64, Option<String>) {
    crate::Program::untyped(input).run()
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
