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
