fn dump_cfg(input: &str) -> String {
    let mut program = crate::Program::typed(input);
    program.parse().unwrap();
    program.check();

    let ctx = program.ctx.unwrap();
    let cfg = program.cfg.unwrap();

    cfg.format(&ctx).unwrap()
}

#[test]
fn capture_basics() {
    let program = r#"
    struct Point {
        x: Int,
        y: Int,
    }
    
    let pt = Point { x: 2, y: 7 }
    fn captures(arg: Int) -> Int {
        ret add(arg, pt.y)
    }

    fn no_captures(arg: Int) -> Int {
        ret add(arg, arg)
    }

    fn calls_funcs() -> Int {
        let a = captures(23)
        let b = no_captures(29)
        ret add(a, b)
    }

    let my_fn = captures
    let my_fn2 = no_captures

    print captures(2)
    print my_fn(3)

    print no_captures(6)
    print my_fn2(7)

    print calls_funcs()

    let x = 1
    let y = true
    let z = (x, y)
    if z.1 {
        print add(1, add(x, 2))
    }
    ret 0
"#;

    let cfg = dump_cfg(program);
    insta::assert_snapshot!("basics", cfg);
}

#[test]
fn loop_basics() {
    let program = r#"
    let cond = true
    let mut counter = 3
    loop {
        if eq(counter, 0) {
            break
        }
        set counter = sub(counter, 1)
        print counter

        loop {
            print "hello"
            if cond {
                break
            } else {
                continue
            }
            print "unreachable"
        }
        print "looped!"
    }
    ret 0
"#;

    let cfg = dump_cfg(program);
    insta::assert_snapshot!("loop_basics", cfg);
}

#[test]
fn func_names_are_paths() {
    let program = r#"
    fn do_a_compy(x: Int) -> Int {
        ret mul(x, x)
    }
    let z = 5
    fn captured_do_a_compy(x: Int) -> Int {
        ret do_a_compy(add(x, z))
    }
    struct MyClosure {
        func: fn(Int) -> Int
        capture: Int
    }
    fn call_closure(close: MyClosure) -> Int {
        ret close.func(close.capture)
    }
    
    let close1 = MyClosure { func: do_a_compy, capture: 7 }
    let close2 = MyClosure { func: captured_do_a_compy, capture: 9 }
    
    print call_closure(close1)
    print call_closure(close2)
    
    ret close2.capture
    "#;

    let cfg = dump_cfg(program);
    insta::assert_snapshot!("func_names_are_paths", cfg);
}
