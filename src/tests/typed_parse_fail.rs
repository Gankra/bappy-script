fn run_typed(input: &str) -> (i64, Option<String>) {
    crate::Program::typed(input).run()
}

#[test]
#[should_panic(expected = "Parse Error")]
fn parse_fail_partial_annotation_1() {
    let program = r#"
        fn foo(x: Int, y: ) {
            ret ()
        }
        ret 0
    "#;

    let (_result, _output) = run_typed(program);
}

#[test]
#[should_panic(expected = "Parse Error")]
fn parse_fail_partial_annotation_2() {
    let program = r#"
        let x: = 2
        ret 0
    "#;

    let (_result, _output) = run_typed(program);
}
