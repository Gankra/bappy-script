fn run_typed(input: &str) -> (i64, Option<String>) {
    crate::Program::typed(input).run()
}

#[test]
fn test_basic() {
    // Just tests basic functionality.
    //
    // Whitespace is wonky to make sure the parser is pemissive of that.
    let program = r#"
        fn square(x:Int)->Int{
            ret mul(x, x)
        }

        let mut x:Int=6
        let cond: Bool = true

        if cond {
            set x = 7
        }

        let y: Int = square(4)
        ret square(x)            
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 49);
}

#[test]
fn test_func_names_are_paths() {
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

    let (result, output) = run_typed(program);
    assert_eq!(result, 9);
    assert_eq!(
        output.unwrap(),
        r#"49
196
"#
    )
}

#[test]
fn test_cfg_bringup_example() {
    let program = r#"
    struct Point {
        x: Int
        y: Int
    }
    
    fn square(x:Int)->Int{
        ret mul(x, x)
    }
    
    let pt = Point { x: 1, y: 3 }
    let z = 4
    
    fn captures() -> Int {
        ret add(square(pt.x), square(pt.y))
    }
    fn super_captures() -> Int {
        ret sub(captures(), z)
    }
    
    print square(9)
    print captures()
    print super_captures()
    ret 1
    "#;

    let (result, output) = run_typed(program);
    assert_eq!(result, 1);
    assert_eq!(
        output.unwrap(),
        r#"81
10
6
"#
    )
}

#[test]
fn test_stack_allocs() {
    let program = r#"
    struct Point { 
        x: Int 
        y: Int
    }
    
    let mut a = 1
    let mut b = true
    let mut c = ()
    let mut d = "hello"
    let mut e = Point { x: 3, y: 5 }
    let mut f = (9, true, 19)
    
    print a
    print b
    print c
    print d
    print e
    print f
    
    set a = 74
    set b = false
    set c = ()
    set d = "bye"
    set e = Point { x: 12, y: 17 }
    set f = (20, false, 45)
    
    print a
    print b
    print c
    print d
    print e
    print f
    
    ret 99
    "#;

    let (result, output) = run_typed(program);
    assert_eq!(result, 99);
    assert_eq!(
        output.unwrap(),
        r#"1
true
()
hello
Point { x: 3, y: 5 }
(9, true, 19)
74
false
()
bye
Point { x: 12, y: 17 }
(20, false, 45)
"#
    )
}

#[test]
fn test_nested_closure_capture() {
    let program = r#"
        let factor: Int = 3
        fn get_factor() -> Int {
            ret factor
        }
        fn multi(x: Int) -> Int {
            ret mul(x, get_factor())
        }

        let x: Int = 7

        ret multi(x)      
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 21);
}

#[test]
fn test_fn_tys() {
    let program = r#"
        let factor: Int = 3
        fn get_factor() -> Int {
            ret factor
        }
        fn multi(factory: fn() -> Int, x: Int) -> Int {
            ret mul(x, factory())
        }

        let x: Int = 7

        print multi(get_factor, x)  


        fn mega_multi(multiplier: fn(fn() -> Int, Int) -> Int) -> Int {
            fn eleven() -> Int {
                ret 11
            }
            print multiplier(eleven, 9)
            ret 0
        }

        let _: Int = mega_multi(multi)

        ret 0
    "#;

    let (result, output) = run_typed(program);
    assert_eq!(result, 0);
    assert_eq!(
        output.unwrap(),
        r#"21
99
"#
    )
}

#[test]
fn test_empty_ty() {
    let program = r#"
        let mut x: () = ()
        set x = ()
        let y: () = x

        fn f1() -> () {
            ret ()
        }
        fn f2(x: ()) -> () {
            ret x
        }
        fn f3(x: (), y: ()) -> () {
            ret y
        }
        fn f4(x: fn() -> ()) -> () {
            ret x()
        }
        fn f5(x: fn(()) -> ()) -> () {
            ret x(())
        }
        fn f6(x: fn(fn() -> ()) -> ()) -> () {
            ret x(f1)
        }
        fn f7(x: fn(fn(()) -> ()) -> ()) -> () {
            ret x(f2)
        }
        fn f8(x: fn(fn((), ()) -> ()) -> (), y: ()) -> () {
            ret x(f3)
        }
        
        let a: () = f1()
        let b: () = f2(())
        let c: () = f3((), ())
        let d: () = f4(f1)
        let e: () = f5(f2)
        let f: () = f6(f4)
        let g: () = f7(f5)
        let h: fn(fn(fn((), ()) -> ()) -> (), ()) -> () = f8

        ret 0           
    "#;

    let (result, _output) = run_typed(program);
    assert_eq!(result, 0);
}

#[test]
fn test_weird_print_ty() {
    // Regression test for that time I accidentally had `print`
    // check that its input was the return type of the parent function.
    let program = r#"
        let factor: Int = 3
        fn get_factor() -> Int {
            ret factor
        }
        fn multi(factory: fn() -> Int, x: Int) -> Int {
            ret mul(x, factory())
        }

        let x: Int = 7

        print multi(get_factor, x)  


        fn mega_multi(multiplier: fn(fn() -> Int, Int) -> Int) -> () {
            fn eleven() -> Int {
                ret 11
            }
            print multiplier(eleven, 9)
            ret ()
        }

        let _: () = mega_multi(multi)

        ret 0
    "#;

    let (result, output) = run_typed(program);
    assert_eq!(result, 0);
    assert_eq!(
        output.unwrap(),
        r#"21
99
"#
    )
}

#[test]
fn test_aggregates_basic() {
    let program = r#"
        let mut factors:(Int,Bool)=( 0, true )
        let factors2: ( Int, Bool ) = (0,true)
        print factors
        set factors = (2, false) 
        print factors

        struct Point {
            x: Int
            y: Str
        }

        let mut pt: Point = Point { x : 0, y : "hello" }
        let pt2:Point=Point{x:0,y:"hello"}
        print pt
        set pt = Point { x: 3, y: "bye" }
        print pt

        ret 0
    "#;

    let (result, output) = run_typed(program);
    assert_eq!(result, 0);
    assert_eq!(
        output.unwrap(),
        r#"(0, true)
(2, false)
Point { x: 0, y: "hello" }
Point { x: 3, y: "bye" }
"#
    )
}

#[test]
fn test_aggregates_captures() {
    let program = r#"
        fn print_1d_point() -> Int {
            struct Point {
                x: Int
            }
            let x: Point = Point { x: 1 }
            print x
            ret 0
        }

        let _:Int = print_1d_point()
        let mut print_point: fn() -> Int = print_1d_point
        let _:Int = print_point()

        if true {
            struct Point {
                x: Int
                y: Int
            }

            fn print_2d_point() -> Int {
                let x: Point = Point { x: 2, y: 4 }
                print x
                ret 0
            }

            let _:Int = print_2d_point();
            set print_point = print_2d_point
        }

        struct Point {
            x: Int
            y: Int
            z: Int
        }

        fn print_3d_point() -> Int {
            let x: Point = Point { x: 3, y: 5, z: 7 }
            print x
            ret 0
        }

        let _:Int = print_1d_point()
        let _:Int = print_point()
        let _:Int = print_3d_point()

        ret 0
    "#;

    let (result, output) = run_typed(program);
    assert_eq!(result, 0);
    assert_eq!(
        output.unwrap(),
        r#"Point { x: 1 }
Point { x: 1 }
Point { x: 2, y: 4 }
Point { x: 1 }
Point { x: 2, y: 4 }
Point { x: 3, y: 5, z: 7 }
"#
    )
}

#[test]
fn test_nominal_shadowing() {
    let program = r#"
        struct Point {
            x: Int,
            y: Int,
            z: Int,
        }

        let x1: Point = Point { x: 1, y: 3, z: 7 }
        let mut x2: Point = Point { x: 2, y: 5, z: 9 }
        print x1
        print x2

        struct Point {
            x: Int,
            y: Int,
        }

        set x2 = x1
        print x1
        print x2

        let y: Point = Point { x: 3, y: 9 }
        print y

        ret 0
    "#;

    let (result, output) = run_typed(program);
    assert_eq!(result, 0);
    assert_eq!(
        output.unwrap(),
        r#"Point { x: 1, y: 3, z: 7 }
Point { x: 2, y: 5, z: 9 }
Point { x: 1, y: 3, z: 7 }
Point { x: 1, y: 3, z: 7 }
Point { x: 3, y: 9 }
"#
    )
}

#[test]
fn test_field_basic() {
    let program = r#"
        struct Point {
            x: Int,
            y: Int,
            z: Int,
        }

        let pt: Point = Point { x: 3, y: 7, z: 12 }
        print pt
        print pt.x
        print pt.y
        print pt.z

        let tup: (Int, Bool, Str) = (19, true, "hello")
        print tup
        print tup.0
        print tup.1
        print tup.2

        fn handle_point(point: Point) -> Int {
            ret add(add(point.x, pt.y), tup.0)
        }

        let pt: Point = Point { x: tup.0, y: add(1, tup.0), z: mul(3, tup.0) }
        print pt
        print pt.x
        print pt.y
        print pt.z

        fn handle_pointer(ptr: fn(Point) -> Int, pt: Point) -> Int {
            ret ptr(pt)
        }

        ret handle_pointer(handle_point, pt)
    "#;

    let (result, output) = run_typed(program);
    assert_eq!(result, 19 + 7 + 19);
    assert_eq!(
        output.unwrap(),
        r#"Point { x: 3, y: 7, z: 12 }
3
7
12
(19, true, "hello")
19
true
hello
Point { x: 19, y: 20, z: 57 }
19
20
57
"#
    )
}

#[test]
fn test_inference_basic() {
    let program = r#"
        struct Point {
            x: Int,
            y: Int,
            z: Int,
        }

        let mut pt = Point { x: 3, y: 7, z: 12 }
        print pt
        print pt.x
        print pt.y
        print pt.z

        let tup = (19, true, "hello")
        print tup
        print tup.0
        print tup.1
        print tup.2

        let pt2 = Point { x: tup.0, y: add(1, tup.0), z: mul(3, tup.0) }
        print pt2
        print pt2.x
        print pt2.y
        print pt2.z

        set pt = pt2

        fn foo(x: Int) {
            ret ()
        }

        let x = foo
        print foo(0)
        print x(2)

        let a = true
        let b = false
        let c = ()
        let d = ""
        let e = "hello"
        let f = 2
        let g = -2

        ret 0
    "#;

    let (result, output) = run_typed(program);
    assert_eq!(result, 0);
    assert_eq!(
        output.unwrap(),
        r#"Point { x: 3, y: 7, z: 12 }
3
7
12
(19, true, "hello")
19
true
hello
Point { x: 19, y: 20, z: 57 }
19
20
57
()
()
"#
    )
}

#[test]
fn test_a_bit_of_everything() {
    let program = r#"
        fn print_1d_point() {
            struct Point {
                x: Int
            }
            let x = Point { x: 1 }
            print x
            ret ()
        }

        let _ = print_1d_point()
        let mut print_point: fn() -> () = print_1d_point
        let _ = print_point()

        let tuple = (1, (true, "hello"), false)
        if tuple.1.0 {
            struct Point {
                x: Int
                y: Int
            }

            let captured_point = Point { x: 2, y: 4 }
            fn print_2d_point() {
                print captured_point
                ret ()
            }

            let _ = print_2d_point();
            set print_point = print_2d_point
        }

        struct Point {
            x: Int
            y: Int
            z: Int
        }

        fn print_3d_point() -> Int {
            let pt: Point = Point { x: 3, y: 5, z: 7 }
            print pt
            ret add(add(pt.x, pt.y), pt.z)
        }

        fn print_many() {
            print "3 more times!!!"
            let mut counter = 3
            loop {
                if eq(counter, 0) {
                    break
                }
                set counter = sub(counter, 1)
                let _ = print_3d_point()
            }
            ret ()
        }

        let _ = print_1d_point()
        let _ = print_point()
        let res = print_3d_point()
        print res
        let _ = print_many()
        ret res
    "#;

    let (result, output) = run_typed(program);
    assert_eq!(result, 15);
    assert_eq!(
        output.unwrap(),
        r#"Point { x: 1 }
Point { x: 1 }
Point { x: 2, y: 4 }
Point { x: 1 }
Point { x: 2, y: 4 }
Point { x: 3, y: 5, z: 7 }
15
3 more times!!!
Point { x: 3, y: 5, z: 7 }
Point { x: 3, y: 5, z: 7 }
Point { x: 3, y: 5, z: 7 }
"#
    )
}

#[test]
fn test_complex_paths() {
    let program = r#"
        struct Point {
            x: Int,
            y: Bool,
        }

        let mut pt = Point { x: 1, y: true }
        print pt
        set pt.x = 3
        print pt
        set pt.y = false
        print pt
        set pt = Point { x: 17, y: true }
        print pt

        let mut tup = (1, "hello", false, ())
        print tup
        set tup.0 = 3
        print tup
        set tup.1 = "bye"
        print tup
        set tup.2 = true
        print tup
        set tup.3 = ()
        print tup

        struct SuperPoint {
            x: (Int, Str)
            y: Bool
        }

        let mut sup = ((SuperPoint { x: (65, "what"), y: false }, 2), 7, ("hello", "there"))
        print sup
        print sup.0.0.x
        print sup.0.0.x.1
        print sup.2.1

        set sup.1 = 5
        print sup
        set sup.0.1 = 3
        print sup
        set sup.2.0 = "bye"
        print sup
        set sup.0.0.x.1 = "wow!"
        print sup
        set sup.0.0.x.0 = add(sup.0.0.x.0, 4)
        print sup

        ret sup.0.0.x.0
    "#;

    let (result, output) = run_typed(program);
    assert_eq!(result, 69);
    assert_eq!(
        output.unwrap(),
        r#"Point { x: 1, y: true }
Point { x: 3, y: true }
Point { x: 3, y: false }
Point { x: 17, y: true }
(1, "hello", false, ())
(3, "hello", false, ())
(3, "bye", false, ())
(3, "bye", true, ())
(3, "bye", true, ())
((SuperPoint { x: (65, "what"), y: false }, 2), 7, ("hello", "there"))
(65, "what")
what
there
((SuperPoint { x: (65, "what"), y: false }, 2), 5, ("hello", "there"))
((SuperPoint { x: (65, "what"), y: false }, 3), 5, ("hello", "there"))
((SuperPoint { x: (65, "what"), y: false }, 3), 5, ("bye", "there"))
((SuperPoint { x: (65, "wow!"), y: false }, 3), 5, ("bye", "there"))
((SuperPoint { x: (69, "wow!"), y: false }, 3), 5, ("bye", "there"))
"#
    )
}

#[test]
fn test_nested_capture() {
    // If an inner closure captures some state outside of their parent,
    // then the parent must also capture that state.
    let program = r#"
        fn temp () -> fn () -> Int {
            fn inner_temp() -> Int {
                ret 2
            }
            ret inner_temp
        }
        let mut func: fn() -> fn () -> Int = temp

        if true {
            let capture = 7
            fn outer_capturer() -> fn () -> Int {
                fn inner_capturer() -> Int {
                    // Nested capture, forcing outer_capturer to capture
                    print capture
                    ret capture
                }
                // Make sure this implicit capture doesn't mess up our
                // ability to define our own variables with that name
                let capture = 12
                print capture
                ret inner_capturer
            }
            set func = outer_capturer
        }
        let sub_func = func()
        ret sub_func()
    "#;

    let (result, output) = run_typed(program);
    assert_eq!(result, 7);
    assert_eq!(
        output.unwrap(),
        r#"12
7
"#
    )
}
