# bappy-script

Just messing around with a toy compiler.

bappy-script is primarily a playground for learning how static type systems work
(checker.rs). The parser and interpreter *work* but are an after-thought by comparison.




# Everything Test Program

This program roughly demonstrates everything the language can do, including:

Basics:

* Basic primitives (Bool, Int, Str, `()`)
* Basic control flow (ret, if, else, loop, break, continue)
* Basic mutable variables (let, set)
* Builtin functions provide add/sub/mul for math and eq/not for conditions

Types:

* First-class functions (which are all closures)
    * Closures capture state by-value at the point where execution reaches their decl.
* Structural product types (tuples)
* Nominal product types (structs)

Analysis:

* Optional Static type checking
    * Types of variables can be inferred because exprs currently always have a known type.
    * Functions must always declare the types of args/returns
        * Although if the return type is omitted it's assumed to be `()`
    * Properly handles shadowing and scoping of nominal types
* Statically validates variable accesses are in scope
* Statically validates control flow (can't `continue` outside a loop)
* Statically computes closure captures

```rust
fn print_1d_point() {
    struct Point {
        x: Int
    }
    let x = Point { x: 1 }
    print x
    ret ()
}

let _ = print_1d_point()
let print_point: fn() -> () = print_1d_point
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
    let counter = 3
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
``` 



# The Parser and Syntax

The parser is bad (brittle) because I just wanted something simple that was easy to extend.
I think it's technically "recursive descent" but I don't like, tokenize so idk. The parser
has the worst errors and no recovery because I just didn't care about it.

Syntax is largely based on Rust's because it's a fairly clean and unambiguous syntax I'm
comfortable with. 

> *Why didn't you make it a lisp variant if you hate parsing?*
>
> I am really bad at reading/writing lispy things so this felt like a good tradeoff
> in personal effort and comfort. It's also easier for me to intuit how something "should"
> work when it looks like Rust code, because that's the language I understand the best.
>
> Also if I tell something that it's Rust code, the syntax highlighting basically just works lol


Notable deviations:

Newlines are very significant. It sucks, but most of the time you don't really notice.
It only really hurts for really complicated expressions and function declarations IMO.
As a consolation I at least let you not write semicolons, since newlines basically are
semicolons?

**Many things must all be contained to their own line:**

* Statements (including all subexprs)
    * `let x: MyType = func1(func2(a, b, c), d)`
    * `ret x`
    * ...
* Block "headers"
    * `struct MyStruct {`
    * `fn (a: Int, b: Bool) -> Int {`
    * `if x {`
    * `} else {`
* Struct fields (`x: Int`)
* Closing braces (`}`)

**Some things are extra verbose because I wanted the parser and interpretter to be trivial:**

* Expressions are not valid statements. So function calls must be part of a larger statement.
    * Typically `let _ = func()` is the easiest way to call a function just for its side-effects.
* Assigning to a pre-existing variable must be prefixed with `set`
    * `set x = y`
* The language isn't expression-oriented, so you must explicitly `set` and `ret` values.

**There are no infix operators**:

* We've got builtins like `add(x, y)` and `eq(x, y)`, final offer





# The Checker (Static Analysis)

This is basically the place I'm most interested in, so you'll find `src/checker.rs`
will have the most comments and discussion. 

Currently all analysis is done directly by recursively walking the AST. All the
variables and types at every point in the program are tracked at every point in
the program, but this information is mutable and transient (sort of like we're
"executing" the program).

Ideally I'd like to write something that can create an (SSA?) Control Flow Graph
to facilitate other analyses like Definite Initialization (a major piece of
the ownership system in Rust, and just a staple of any good compiler because
it lets you report things like "value assigned to variable is never used").

I'm potentially also interested in faffing around with Generics and *maybe*
Higher-Rank Types? But I'm genuinely unsure about how to best represent and
implement some of that stuff (or rather, am saddened that type comparison might
have to be more complex than comparing type ids?).

To reiterate the notes in the example, the type system currently supports:

Basics:

* Basic primitives (Bool, Int, Str, `()`)
* Basic control flow (ret, if, else, loop, break, continue)
* Basic mutable variables (let, set)

Types:

* First-class functions (which are all closures)
    * Closures capture state by-value at the point where execution reaches their decl.
* Structural product types (tuples)
* Nominal product types (structs)

Analysis:

* Optional Static type checking
    * Types of variables can be inferred because exprs currently always have a known type.
    * Functions must always declare the types of args/returns
        * Although if the return type is omitted it's assumed to be `()`
    * Properly handles shadowing and scoping of nominal types
* Statically validates variable accesses are in scope
* Statically validates control flow (can't `continue` outside a loop)
* Statically computes closure captures




# The Interpreter

It's a pretty simple "untyped" interpreter of the AST. Runtime values do have basic type
tagging so we can check if something is a function or boolean for the places where
only those types make sense.

It *almost* doesn't depend at all on the checker, but the closure capture analysis
adds some required information to the AST.

Not depending on the checker helps catch bugs in the checker.

Nothing is terribly optimized. Any time a value is semantically moved it's Cloned
(and things like tuples, closures, and structs all contain Vecs!). But even given
that, compiling and evaluating the \~100 programs that are currently in the tests
is basically instantaneous on my beefy work machine.

There's a lot more lifetimes in the interpreter than there really should be.
I make a point of keeping all string literals as pointers into the original
program text, because that was aesthetically important to me.