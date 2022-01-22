fn run(input: &str) -> (i64, Option<String>) {
    crate::Program::untyped(input).run()
}

#[test]
fn test_factorial() {
    let program = r#"
        fn factorial(self, val) {
            if eq(val, 0) {
                ret 1
            }
            ret mul(self(self, sub(val, 1)), val)
        }

        print factorial(factorial, 0)
        print factorial(factorial, 1)
        print factorial(factorial, 2)
        print factorial(factorial, 3)
        print factorial(factorial, 4)
        print factorial(factorial, 5)
        print factorial(factorial, 6)
        print factorial(factorial, 7)

        ret 0
    "#;

    let (result, output) = run(program);
    assert_eq!(result, 0);
    assert_eq!(
        output.unwrap(),
        r#"1
1
2
6
24
120
720
5040
"#
    )
}

#[test]
fn test_comments() {
    let program = r#"
        // Hello!
        let x = 0
        // let x = 2
        // print "fuck!"
        print "yay"
        
        let whatever = 9

        // fn whatever() {
        //   ret 7
        // }
        //

        // }

        print whatever

        // ret -1
        ret x
    "#;

    let (result, output) = run(program);
    assert_eq!(result, 0);
    assert_eq!(
        output.unwrap(),
        r#"yay
9
"#
    );
}

#[test]
fn test_if() {
    let program = r#"
        let x = true
        let y = 2

        fn captures() {
            if x {
                ret y
            }
        }

        fn False() {
            ret false
        }

        if x {
            print "yes1"
            print y
        }
        
        print "normal1"
        
        if False() {
            print "oh no!"
            ret -2
        } else {
            print "else1"
        }

        if False() {
            print "oh no!"
        }

        print "normal2"

        let x = false
        let y = 3
        print captures()
        print x
        print y

        fn captures2() {
            if x {
                ret 1
            } else {
                ret sub(y, 4)
            }
        }

        let x = 1
        let y = 4
        print captures2()
        print x
        print y

        if true {
            print "yes2"
            ret 999
        }

        ret -1
    "#;

    let (result, output) = run(program);
    assert_eq!(result, 999);
    assert_eq!(
        output.unwrap(),
        r#"yes1
2
normal1
else1
normal2
2
false
3
-1
1
4
yes2
"#
    );
}

#[test]
fn test_builtin_bool() {
    let program = r#"
        let x = 0
        if eq(x, 0) {
            print "eq!"
        }
        if not(eq(x, 1)) {
            print "neq!"
        }

        if eq(x, 1) {
            ret -1
        }
        ret 0
    "#;

    let (result, output) = run(program);
    assert_eq!(result, 0);
    assert_eq!(
        output.unwrap(),
        r#"eq!
neq!
"#
    );
}

#[test]
fn test_builtin_math() {
    let program = r#"
        ret sub(mul(add(4, 7), 13), 9)
    "#;

    let (result, _output) = run(program);
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

    let (result, _output) = run(program);
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

    let (result, _output) = run(program);
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

    let (result, _output) = run(program);
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

    let (result, _output) = run(program);
    assert_eq!(result, 66);
}

#[test]
fn test_call_capture() {
    let program = r#"
        fn double(x) {
            fn two(val) {
                ret x(x(val))
            }
            ret two
        }

        fn succ(x) {
            ret add(x, 1)
        }

        let add_two = double(succ)
        let add_four = double(add_two)
        let add_eight = double(add_four)

        let a = add_two(1)
        let b = add_four(1)
        let c = add_eight(1)

        ret add(add(a, b), c)
    "#;

    let (result, _output) = run(program);
    assert_eq!(result, 17);
}

#[test]
fn test_negative() {
    let program = r#"
        ret -1
    }
    "#;

    let (result, _output) = run(program);
    assert_eq!(result, -1);
}

#[test]
fn test_fake_bools() {
    let program = r#"
        fn True(if, else) {
            ret if()
        }
        fn False(if, else) {
            ret else()
        }

        let condition = True
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

    let (result, output) = run(program);
    assert_eq!(result, 70);
    assert_eq!(output.unwrap(), "1\n");
}

#[test]
fn test_idents() {
    let program = r#"
        let _x = 66
        let __y = 55
        let _0 = 44
        let _x_y__z_ = 33
        ret add(add(add(_x, __y), _0), _x_y__z_)
    "#;

    let (result, _output) = run(program);
    assert_eq!(result, 66 + 55 + 44 + 33);
}

#[test]
fn test_literals() {
    let program = r#"
        let a = 66
        let b = -55
        let c = "hello"
        let d = ""
        let e = true
        let f = false
        let g = ()

        print a
        print b
        print c
        print d
        print e
        print f
        print g

        ret a
    "#;

    let (result, output) = run(program);
    assert_eq!(result, 66);
    assert_eq!(
        output.unwrap(),
        r#"66
-55
hello

true
false
()
"#
    );
}

#[test]
fn test_str() {
    let program = r#"
        print "hello"
        ret 1
    "#;

    let (result, output) = run(program);
    assert_eq!(result, 1);
    assert_eq!(output.unwrap(), "hello\n");
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

    let (result, _output) = run(program);
    assert_eq!(result, 99);
}

#[test]
fn test_loops_no_looping() {
    let program = r#"
        loop {
            if true {
                print "yay1"
                break
            } else {
                continue
            }
        }

        let x = 17
        fn do_stuff() {
            loop {
                print x
                ret 2
            }
        }

        let x = 12
        ret do_stuff()
    "#;

    let (result, output) = run(program);
    assert_eq!(result, 2);
    assert_eq!(
        output.unwrap(),
        r#"yay1
17
"#
    );
}

#[test]
fn test_loops_and_set() {
    let program = r#"
        let mut x = 10

        fn remembers_original() {
            ret x
        }
        loop {
            fn remembers_previous() {
                ret x
            }

            // Exit the loop at 0
            if eq(x, 0) {
                break
            }
            set x = sub(x, 1)

            // Skip 2, no one likes 2
            if eq(x, 2) {
                continue
            }

            print "loop!"
            print remembers_original()
            print remembers_previous()
            print x
        }

        ret 0
    "#;

    let (result, output) = run(program);
    assert_eq!(result, 0);
    assert_eq!(
        output.unwrap(),
        r#"loop!
10
10
9
loop!
10
9
8
loop!
10
8
7
loop!
10
7
6
loop!
10
6
5
loop!
10
5
4
loop!
10
4
3
loop!
10
2
1
loop!
10
1
0
"#
    );
}

#[test]
fn test_set_basic() {
    let program = r#"
        let mut x = 0
        print x
        
        set x = 3
        print x
        
        set x = add(x, 8)
        print x
        
        if true {
            set x = 27
        } else {
            set x = 4
        }
        print x

        if false {
            set x = 2
        } else {
            set x = 35
        }
        print x

        // reinitialize value
        let mut x = 58
        print x

        set x = 71
        print x

        ret 0
    "#;

    let (result, output) = run(program);
    assert_eq!(result, 0);
    assert_eq!(
        output.unwrap(),
        r#"0
3
11
27
35
58
71
"#
    );
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
        // This is the key line we're really testing
        ret close.func(close.capture)
    }
    
    let close1 = MyClosure { func: do_a_compy, capture: 7 }
    let close2 = MyClosure { func: captured_do_a_compy, capture: 9 }
    
    print call_closure(close1)
    print call_closure(close2)
    
    ret close2.capture
    "#;

    let (result, output) = run(program);
    assert_eq!(result, 9);
    assert_eq!(
        output.unwrap(),
        r#"49
196
"#
    )
}

#[test]
fn test_func_path_capture() {
    let program = r#"
    let y = 3
    fn normal(x: Int) -> Bool {
        ret eq(x, 24)
    }
    fn captures(x: Int) -> Int {
        ret mul(x, y)
    }
    let tup = (normal, captures)

    fn captures_tup() -> Int {
        // Check that these function calls trigger the vars to be captured
        let cond = tup.0(24)
        let b = tup.1(7)

        if cond {
            print b
        }
        ret b
    }

    ret captures_tup()
    "#;

    let (result, output) = run(program);
    assert_eq!(result, 21);
    assert_eq!(
        output.unwrap(),
        r#"21
"#
    )
}

#[test]
fn test_aggregates_basic() {
    let program = r#"
        let mut factors: (Int, Bool) = (0, true)
        print factors
        set factors = (2, false)
        print factors

        struct Point {
            x: Int
            y: Int
        }

        let mut pt: Point = Point { x: 0, y: 1 }
        print pt
        set pt = Point { x: 3, y: 4 }
        print pt

        ret 0
    "#;

    let (result, output) = run(program);
    assert_eq!(result, 0);
    assert_eq!(
        output.unwrap(),
        r#"(0, true)
(2, false)
Point { x: 0, y: 1 }
Point { x: 3, y: 4 }
"#
    )
}

#[test]
fn test_aggregates_anon_structs() {
    // In untyped mode, we can use structs without declaring them,
    // and change the type of aggregates as we please.
    let program = r#"
        let mut factors = (0, true)
        print factors
        set factors = (2, "bye", ())
        print factors

        let mut pt = Point { x: 0, y: 1 }
        print pt
        set pt = Point { x: 3, y: true, z: "hello" }
        print pt

        ret 0
    "#;

    let (result, output) = run(program);
    assert_eq!(result, 0);
    assert_eq!(
        output.unwrap(),
        r#"(0, true)
(2, "bye", ())
Point { x: 0, y: 1 }
Point { x: 3, y: true, z: "hello" }
"#
    )
}
