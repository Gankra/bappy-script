fn run(input: &str) -> (i64, Option<String>) {
    crate::Program::untyped(input).run()
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
