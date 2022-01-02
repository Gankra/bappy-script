fn run_typed(input: &str) -> (i64, Option<String>) {
    crate::Program::typed(input).run()
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_bad_field_1() {
    let program = r#"
        let x = true
        ret x.0
    "#;

    let (_result, _output) = run_typed(program);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_bad_field_2() {
    let program = r#"
        let x = (0, 1)
        ret x.2
    "#;

    let (_result, _output) = run_typed(program);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_bad_field_3() {
    let program = r#"
        struct Point {
            x: Int
            y: Bool
        }
        let x = Point { x: 1, y: true }
        ret x.z
    "#;

    let (_result, _output) = run_typed(program);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_bad_field_4() {
    let program = r#"
        struct Point {
            x: Int
            y: Bool
        }
        let x = Point { x: 1, y: true }
        ret x.0
    "#;

    let (_result, _output) = run_typed(program);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_inferred_types_enforced_1() {
    let program = r#"
        struct Point {
            x: Int
            y: Bool
        }
        let x = Point { x: 1, y: true }
        set x = true
        ret 0
    "#;

    let (_result, _output) = run_typed(program);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_inferred_types_enforced_2() {
    let program = r#"
        fn foo() {
            ret 1
        }
        let _ = foo()
        ret 0
    "#;

    let (_result, _output) = run_typed(program);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_args_must_have_types() {
    let program = r#"
        fn foo(x: Int, y) {
            ret ()
        }
        let _ = foo
        ret 0
    "#;

    let (_result, _output) = run_typed(program);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_evil_nominal_smuggling_complex() {
    // If you don't handle nested nominal types deeply, then you can get
    // situations where the type changes from underneath you!
    let program = r#"
        struct Point {
            x: Int,
            y: Int,
            z: Int,
        }

        fn handle_point(point: Point) -> Point {
            print "innocent"
            print point
            ret point
        }

        fn handle_pointer(ptr: fn(Point) -> Point, pt: Point) -> Int {
            let pt2: Point = ptr(pt)
            print pt2
            ret 0
        }

        let f: fn(Point) -> Point = handle_point

        let _: Int = handle_pointer(f, Point { x: 1, y: 3, z: 7 })

        struct Point {
            x: Int,
            y: Int,
        }

        fn handle_point_2(point: Point) -> Point {
            print "evil"
            print point
            ret point
        }

        set f = handle_point_2

        let _: Int = handle_pointer(f, Point { x: 2, y: 5 })

        ret 0
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 0);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_evil_nominal_smuggling_simple_1() {
    // If you don't handle shadowed nominal types, then you can get
    // situations where the type changes from underneath you!
    let program = r#"
        struct Point {
            x: Int,
            y: Int,
            z: Int,
        }

        let x: Point = Point { x: 1, y: 3, z: 7 }
        print x

        struct Point {
            x: Int,
            y: Int,
        }
        set x = Point { x: 2, y: 5 }
        print x

        ret 0
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 0);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_evil_nominal_smuggling_simple_2() {
    // If you don't handle nested nominal types deeply, then you can get
    // situations where the type changes from underneath you!
    let program = r#"
        struct Point {
            x: Int,
            y: Int,
            z: Int,
        }

        let x: (Point, Point) = (Point { x: 1, y: 3, z: 7 }, Point { x: 2, y: 5, z: 8})
        print x

        struct Point {
            x: Int,
            y: Int,
        }
        set x = (Point { x: 2, y: 5 }, Point { x: 3, y: 4 })
        print x

        ret 0
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 0);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_evil_nominal_smuggling_simple_3() {
    // If you don't handle nested nominal types deeply, then you can get
    // situations where the type changes from underneath you!
    let program = r#"

        struct Point {
            x: Int,
            y: Int,
            z: Int,
        }
        struct MyTuple {
            a: Point,
            b: Point,
        }

        let x: MyTuple = MyTuple { a: Point { x: 1, y: 3, z: 7 }, b: Point { x: 2, y: 5, z: 8}}
        print x

        struct Point {
            x: Int,
            y: Int,
        }
        set x = MyTuple { a: Point { x: 2, y: 5 }, b: Point { x: 3, y: 4 } }
        print x

        ret 0
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 0);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_undefined_struct_1() {
    let program = r#"
        let pt: Point = Point { x: 0, y: 1 }
        print pt
        set pt = Point { x: 3, y: 4 }
        print pt

        ret 0
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 0);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_undefined_struct_2() {
    let program = r#"
        if true {
            struct Point {
                x: Int
                y: Int
            } 
        }
        let pt: Point = Point { x: 0, y: 1 }
        print pt
        set pt = Point { x: 3, y: 4 }
        print pt

        ret 0
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 0);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_undefined_struct_3() {
    let program = r#"
        let pt: Point = Point { x: 0, y: 1 }
        print pt
        set pt = Point { x: 3, y: 4 }
        print pt

        struct Point {
            x: Int
            y: Int                
        }

        ret 0
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 0);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_undefined_struct_4() {
    let program = r#"
        fn get_point() -> Point {
            struct Point {
                x: Int
                y: Int                
            }
            ret Point { x: 0, y: 0 }
        }
        print get_point()

        ret 0
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 0);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_undefined_struct_5() {
    let program = r#"
        fn get_point(pt: Point) {
            struct Point {
                x: Int
                y: Int                
            }
            ret pt
        }
        print get_point()

        ret 0
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 0);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_bad_struct_field_name() {
    let program = r#"
        struct Point {
            x: Int
            y: Int
        } 

        let pt: Point = Point { x: 0, y: 1 }
        print pt
        set pt = Point { x: 3, z: 4 }
        print pt

        ret 0
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 0);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_bad_struct_field_order() {
    let program = r#"
        struct Point {
            x: Int
            y: Int
        } 

        let pt: Point = Point { x: 0, y: 1 }
        print pt
        set pt = Point { y: 4, x: 3 }
        print pt

        ret 0
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 0);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_bad_struct_field_type() {
    let program = r#"
        struct Point {
            x: Int
            y: Int
        } 

        let pt: Point = Point { x: 0, y: 1 }
        print pt
        set pt = Point { x: 3, y: true }
        print pt

        ret 0
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 0);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_bad_struct_field_count_1() {
    let program = r#"
        struct Point {
            x: Int
            y: Int
        } 

        let pt: Point = Point { x: 0, y: 1 }
        print pt
        set pt = Point { x: 3 }
        print pt

        ret 0
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 0);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_bad_struct_field_count_2() {
    let program = r#"
        struct Point {
            x: Int
            y: Int
        } 

        let pt: Point = Point { x: 0, y: 1 }
        print pt
        set pt = Point { x: 3, y: 2, z: 3 }
        print pt

        ret 0
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 0);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_bad_struct_dupe_field_1() {
    let program = r#"
        struct Point {
            x: Int
            y: Int
        } 

        let pt: Point = Point { x: 0, y: 1 }
        print pt
        set pt = Point { x: 3, x: 2, y: 3 }
        print pt

        ret 0
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 0);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_bad_struct_dupe_field_2() {
    let program = r#"
        struct Point {
            x: Int
            y: Int
        } 

        let pt: Point = Point { x: 0, y: 1 }
        print pt
        set pt = Point { x: 3, x: 2 }
        print pt

        ret 0
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 0);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_bad_tuple_length_1() {
    let program = r#"
        let factors: (Int, Bool) = (0, true)
        print factors
        set factors = (2, false, 3)
        print factors

        ret 0
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 0);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_bad_tuple_length_2() {
    let program = r#"
        let factors: (Int, Bool, Int) = (0, true)
        print factors
        set factors = (2, false)
        print factors

        ret 0
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 0);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_bad_tuple_ty() {
    let program = r#"
        let factors: (Int, Bool) = (0, true)
        print factors
        set factors = (2, 2)
        print factors

        ret 0
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 0);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_scoping_1() {
    let program = r#"
        if false {
            let factor: Int = 3
        }
        fn do_thing() -> Int {
            ret factor
        }
        ret do_thing()
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 3);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_scoping_2() {
    let program = r#"
        if true {
            let factor: Int = 3
        }
        fn do_thing() -> Int {
            ret factor
        }
        ret do_thing()
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 3);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_scoping_3() {
    let program = r#"
        if true {
            let factor: Int = 3
        }
        set factor = 4
        ret factor
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 4);
}

#[test]
#[should_panic(expected = "Compile Error")]
fn compile_fail_set_capture() {
    let program = r#"
        let captured: Int = 3
        fn captures_state() {
            set captured = 4
        }
        ret captured
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 4);
}

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
#[should_panic(expected = "Compile Error")]
fn compile_fail_capture_ty() {
    let program = r#"
        let factor: Bool = true
        fn multi(x: Int) -> Int {
            ret mul(x, factor)
        }

        let x: Int = 7

        ret multi(x)
    "#;

    let (_result, _output) = run_typed(program);
}
