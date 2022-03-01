//! Static analysis! Type checking, name resolution, compilation, ...
//!
//! Much of the analysis done here is optional and ignored by the interpretter.
//! Having things get checked at runtime helps catch compiler bugs!
//!
//! `Program.typed` configures whether we do static type checking (enabled by default).
//!
//! `fn check` is the entry point, although `struct TyCtx` is the "brains" of the type system.
//!
//! The one required thing we do right now is telling all the functions
//! what there captures are, so that the intepretter knows what to capture into
//! the functions

use crate::*;

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;

/// The structure of a type, with all subtypes resolved to type ids (TyIdx).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ty<'p> {
    Int,
    Str,
    Bool,
    Box,
    TypedPtr(TyIdx),
    Empty,
    Func {
        arg_tys: Vec<TyIdx>,
        return_ty: TyIdx,
    },
    Tuple(Vec<TyIdx>),
    NamedStruct(StructTy<'p>),
    Unknown,
}

/// The Ty of a Struct.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructTy<'p> {
    pub name: &'p str,
    pub fields: Vec<FieldTy<'p>>,
}

/// The Ty of a specific field of a struct.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldTy<'p> {
    pub ident: &'p str,
    pub ty: TyIdx,
}

/// A canonical type id.
pub type TyIdx = usize;

/// Information on all the types.
///
/// The key function of TyCtx is to `memoize` all parsed types (TyName) into
/// type ids (TyIdx), to enable correct type comparison. Two types are equal
/// *if and only if* they have the same TyIdx.
///
/// This is necessary because *nominal* types (TyName::Named, i.e. structs) can
/// be messy due to shenanigans like captures/scoping/shadowing/inference. Types
/// may refer to names that are out of scope, and two names that are equal
/// (as strings) may not actually refer to the same type declaration.
///
/// To handle this, whenever a new named type is declared ([push_struct_decl][]),
/// we generate a unique type id (TyIdx) for it. Then whenever we encounter
/// a reference to a Named type, we lookup the currently in scope TyIdx for that
/// name, and use that instead. Named type scoping is managed by `envs`.
///
/// Replacing type names with type ids requires a change of representation,
/// which is why we have [Ty][]. A Ty is the *structure* of a type with all subtypes
/// resolved to TyIdx's (e.g. a field of a tuple, the return type of a function).
/// For convenience, non-typing metadata may also be stored in a Ty.
///
/// So a necessary intermediate step of converting TyName to a TyIdx is to first
/// convert it to a Ty. This intermediate value is stored in `tys`.
/// If you have a TyIdx, you can get its Ty with [realize_ty][]. This lets you
/// e.g. check if a value being called is actually a Func, and if it is,
/// what the type ids of its arguments/return types are.
///
/// `ty_map` stores all the *structural* Tys we've seen before (everything that
/// *isn't* TyName::Named), ensuring two structural types have the same TyIdx.
/// i.e. `(Bool, Int)` will have the same TyIdx everywhere it occurs.
#[derive(Debug)]
pub struct TyCtx<'p> {
    /// Whether static types are enabled/enforced.
    pub is_typed: bool,
    /// The list of every known type.
    ///
    /// These are the "canonical" copies of each type. Types are
    /// registered here via `memoize`, which returns a TyIdx into
    /// this array.
    ///
    /// Types should be compared by checking if they have the same
    /// TyIdx. This allows you to properly compare nominal types
    /// in the face of shadowing and similar situations.
    pub tys: Vec<Ty<'p>>,
    pub needs_drop: Vec<bool>,

    /// Mappings from structural types we've seen to type indices.
    ///
    /// This is used to get the canonical TyIdx of a structural type
    /// (including builtin primitives).
    ///
    /// Nominal types (structs) are stored in `envs`, because they
    /// go in and out of scope.
    ty_map: HashMap<Ty<'p>, TyIdx>,

    /// Scoped type info, reflecting the fact that struct definitions
    /// and variables come in and out of scope.
    ///
    /// These values are "cumulative", so type names and variables
    /// should be looked up by searching backwards in this array.
    ///
    /// If nothing is found, that type name / variable name is undefined
    /// at this point in the program.
    envs: Vec<CheckEnv<'p>>,
    /// Builtin primitive types, for easy lookup.
    pub ty_unknown: TyIdx,
    pub ty_empty: TyIdx,
    pub ty_bool: TyIdx,
    pub ty_int: TyIdx,
    pub ty_ptr: TyIdx,
}

/// Information about types for a specific scope.
#[derive(Debug)]
struct CheckEnv<'p> {
    /// The types of variables
    var_map: HashMap<&'p str, usize>,
    vars: Vec<Var>,
    /// The struct definitions and TyIdx's
    tys: HashMap<&'p str, TyIdx>,

    /// The kind of block this environment was introduced by.
    ///
    /// This is important for analyzing control flow and captures.
    /// e.g. you need to know what scope a `break` or `continue` refers to.
    block_kind: BlockKind,
}

impl<'p> CheckEnv<'p> {
    fn push_var(&mut self, name: &'p str, var: Var) {
        self.var_map.insert(name, self.vars.len());
        self.vars.push(var);
    }
    /*
        fn get_var(&str) ->  {

        }
    */
}

/// Kinds of blocks, see CheckEnv
#[derive(Debug)]
enum BlockKind {
    Func,
    Loop,
    If,
    Else,
    /// Just a random block with no particular semantics other than it being
    /// a scope for variables.
    General,
}

#[derive(Debug, Clone)]
pub struct Reg {
    ty: TyIdx,
    reg: RegIdx,
}

#[derive(Debug, Clone)]
pub enum Var {
    /// A variable that is an alloca (*T)
    Alloca { ty: TyIdx, reg: RegIdx },
    /// A variable that is a value (T)
    Reg { ty: TyIdx, reg: RegIdx },
    /// A variable that is a global function pointer
    GlobalFunc {
        ty: TyIdx,
        global_func: GlobalFuncIdx,
    },
}

impl Var {
    fn alloca(ty: TyIdx, reg: RegIdx) -> Self {
        Self::Alloca { ty, reg }
    }

    fn reg(ty: TyIdx, reg: RegIdx) -> Self {
        Self::Reg { ty, reg }
    }

    fn global_func(ty: TyIdx, global_func: GlobalFuncIdx) -> Self {
        Self::GlobalFunc { ty, global_func }
    }

    pub fn ty(&self) -> TyIdx {
        match *self {
            Var::Alloca { ty, .. } => ty,
            Var::Reg { ty, .. } => ty,
            Var::GlobalFunc { ty, .. } => ty,
        }
    }

    fn needs_temp(&self) -> bool {
        match self {
            Var::Alloca { .. } => true,
            Var::Reg { .. } => false,
            Var::GlobalFunc { .. } => true,
        }
    }
}

impl Reg {
    fn new(ty: TyIdx, reg: RegIdx) -> Self {
        Self { ty, reg }
    }
}

/// The result of a resolve_var lookup, allowing the value (type) to be set.
struct ResolvedVar<'a> {
    /// How deep the capture was.
    /// 0 = local (no capture)
    /// 1 = captured from parent function
    /// 2 = captured from grandparent function (parent must now capture it too)
    /// ...etc
    capture_depth: usize,
    /// The variable
    entry: &'a mut Var,
}

impl<'p> TyCtx<'p> {
    /// Resolve a variable name (its type) at this point in the program.
    fn resolve_var(&mut self, var_name: &'p str) -> Option<ResolvedVar> {
        // By default we're accessing locals
        let mut capture_depth = 0;
        for env in self.envs.iter_mut().rev() {
            if let Some(entry) = env.var_map.get(var_name).map(|&idx| &mut env.vars[idx]) {
                return Some(ResolvedVar {
                    capture_depth,
                    entry,
                });
            }
            if let BlockKind::Func = env.block_kind {
                // We're walking over a function root, so we're now
                // accessing captures. We track this with an integer
                // because if we capture multiple functions deep, our
                // ancestor functions also need to capture that value.
                capture_depth += 1;
            }
        }
        None
    }

    /// Register a new nominal struct in this scope.
    fn push_struct_decl(
        &mut self,
        program: &mut Program<'p>,
        struct_decl: StructDecl<'p>,
    ) -> TyIdx {
        let fields = struct_decl
            .fields
            .iter()
            .map(|f| FieldTy {
                ident: f.ident,
                ty: self.memoize_ty(program, &f.ty),
            })
            .collect::<Vec<_>>();
        let ty_idx = self.tys.len();
        let needs_drop = fields.iter().any(|field| self.needs_drop[field.ty]);
        self.tys.push(Ty::NamedStruct(StructTy {
            name: struct_decl.name,
            fields,
        }));
        self.needs_drop.push(needs_drop);
        self.envs
            .last_mut()
            .unwrap()
            .tys
            .insert(struct_decl.name, ty_idx);
        ty_idx
    }

    /// Resolve the type id (TyIdx) associated with a nominal type (struct name),
    /// at this point in the program.
    fn resolve_nominal_ty<'a>(&'a mut self, ty_name: &'p str) -> Option<TyIdx> {
        if self.is_typed {
            for (_depth, env) in self.envs.iter_mut().rev().enumerate() {
                if let Some(ty) = env.tys.get(ty_name) {
                    return Some(*ty);
                }
            }
            None
        } else {
            None
        }
    }

    /// Converts a TyName (parsed type) into a TyIdx (type id).
    ///
    /// All TyNames in the program must be memoized, as this is the only reliable
    /// way to do type comparisons. See the top level docs of TyIdx for details.
    fn memoize_ty(&mut self, program: &mut Program<'p>, ty_name: &TyName<'p>) -> TyIdx {
        if self.is_typed {
            match ty_name {
                TyName::Int => self.memoize_inner(Ty::Int),
                TyName::Str => self.memoize_inner(Ty::Str),
                TyName::Bool => self.memoize_inner(Ty::Bool),
                TyName::Empty => self.memoize_inner(Ty::Empty),
                TyName::Unknown => self.memoize_inner(Ty::Unknown),
                TyName::Func { arg_tys, return_ty } => {
                    let arg_tys = arg_tys
                        .iter()
                        .map(|arg_ty_name| self.memoize_ty(program, arg_ty_name))
                        .collect();
                    let mut return_ty = self.memoize_ty(program, return_ty);

                    // Allow the return type to be elided
                    if return_ty == self.ty_unknown {
                        return_ty = self.ty_empty;
                    }
                    self.memoize_inner(Ty::Func { arg_tys, return_ty })
                }
                TyName::Tuple(arg_ty_names) => {
                    let arg_tys = arg_ty_names
                        .iter()
                        .map(|arg_ty_name| self.memoize_ty(program, arg_ty_name))
                        .collect::<Vec<_>>();
                    self.memoize_inner(Ty::Tuple(arg_tys))
                }
                TyName::Named(name) => {
                    // Nominal types take a separate path because they're scoped
                    if let Some(ty_idx) = self.resolve_nominal_ty(name) {
                        ty_idx
                    } else {
                        // FIXME: rejig this so the line info is better
                        program.error(
                            format!("Compile Error: use of undefined type name: {}", name),
                            Span {
                                start: addr(program.input),
                                end: addr(program.input),
                            },
                        )
                    }
                }
            }
        } else {
            0
        }
    }

    /// Converts a Ty (structural type with all subtypes resolved) into a TyIdx (type id).
    fn memoize_inner(&mut self, ty: Ty<'p>) -> TyIdx {
        if let Some(idx) = self.ty_map.get(&ty) {
            *idx
        } else {
            let needs_drop = match &ty {
                Ty::Int => false,
                Ty::Str => false,
                Ty::Bool => false,
                Ty::Box => true,
                Ty::TypedPtr(_) => false,
                Ty::Empty => false,
                Ty::Func { .. } => true,
                Ty::Tuple(arg_tys) => arg_tys.iter().any(|&ty| self.needs_drop[ty]),
                Ty::NamedStruct(_) => unreachable!(),
                Ty::Unknown => false,
            };

            let ty1 = ty.clone();
            let ty2 = ty;
            let idx = self.tys.len();

            self.ty_map.insert(ty1, idx);
            self.tys.push(ty2);
            self.needs_drop.push(needs_drop);
            idx
        }
    }

    /// Get the type-structure (Ty) associated with this type id (TyIdx).
    pub fn realize_ty(&self, ty: TyIdx) -> &Ty<'p> {
        if self.is_typed {
            self.tys
                .get(ty)
                .expect("Internal Compiler Error: invalid TyIdx")
        } else {
            &Ty::Unknown
        }
    }

    pub fn pointee_ty(&self, ty: TyIdx) -> TyIdx {
        if let Ty::TypedPtr(pointee) = self.realize_ty(ty) {
            *pointee
        } else {
            unreachable!("expected typed to be pointer");
        }
    }

    /// Stringify a type.
    pub fn format_ty(&self, ty: TyIdx) -> String {
        match self.realize_ty(ty) {
            Ty::Int => format!("Int"),
            Ty::Str => format!("Str"),
            Ty::Bool => format!("Bool"),
            Ty::Empty => format!("()"),
            Ty::Unknown => format!("<unknown>"),
            Ty::Box => format!("Box"),
            Ty::TypedPtr(pointee_ty) => {
                let pointee = self.format_ty(*pointee_ty);
                format!("&{}", pointee)
            }

            Ty::NamedStruct(struct_decl) => format!("{}", struct_decl.name),
            Ty::Tuple(arg_tys) => {
                let mut f = String::new();
                write!(f, "(").unwrap();
                for (idx, arg_ty_idx) in arg_tys.iter().enumerate() {
                    if idx != 0 {
                        write!(f, ", ").unwrap();
                    }
                    let arg = self.format_ty(*arg_ty_idx);
                    write!(f, "{}", arg).unwrap();
                }
                // Try to visually distinguish the single-element-tuple
                if arg_tys.len() == 1 {
                    write!(f, ",").unwrap();
                }
                write!(f, ")").unwrap();
                f
            }
            Ty::Func { arg_tys, return_ty } => {
                let mut f = String::new();
                write!(f, "fn (").unwrap();
                for (idx, arg_ty_idx) in arg_tys.iter().enumerate() {
                    if idx != 0 {
                        write!(f, ", ").unwrap();
                    }
                    let arg = self.format_ty(*arg_ty_idx);
                    write!(f, "{}", arg).unwrap();
                }
                write!(f, ") -> ").unwrap();

                let ret = self.format_ty(*return_ty);
                write!(f, "{}", ret).unwrap();
                f
            }
        }
    }
}

impl<'p> Program<'p> {
    /// Type check the program, including some mild required "compilation".
    ///
    /// The "compilation" is just computing each closure's capture set, which
    /// the runtime needs to know to save the relevant state when it finds a
    /// function decl.
    pub fn check(&mut self) {
        let mut ctx = TyCtx {
            tys: Vec::new(),
            needs_drop: Vec::new(),
            ty_map: HashMap::new(),
            envs: Vec::new(),
            is_typed: self.typed,
            ty_unknown: 0,
            ty_empty: 0,
            ty_ptr: 0,
            ty_int: 0,
            ty_bool: 0,
        };

        let mut cfg = Cfg {
            funcs: Vec::new(),
            nominals: Vec::new(),
            func_stack: Vec::new(),
            main: GlobalFuncIdx(0), // Dummy value
        };

        // Cache some key types
        ctx.ty_unknown = ctx.memoize_inner(Ty::Unknown);
        ctx.ty_empty = ctx.memoize_inner(Ty::Empty);
        ctx.ty_bool = ctx.memoize_inner(Ty::Bool);
        ctx.ty_int = ctx.memoize_inner(Ty::Int);
        ctx.ty_ptr = ctx.memoize_inner(Ty::Box);

        // Set up globals (stdlib)
        let mut globals = CheckEnv {
            vars: Vec::new(),
            var_map: HashMap::new(),
            tys: HashMap::new(),
            // Doesn't really matter what this value is for the globals
            block_kind: BlockKind::General,
        };
        for builtin in self.builtins.clone() {
            let global_ty = ctx.memoize_ty(self, &builtin.ty);
            let new_global = cfg.push_global_func(builtin.name, builtin.args.to_owned(), global_ty);
            // Intrinsics have no bodies, pop them immediately
            cfg.pop_global_func();
            globals.push_var(builtin.name, Var::global_func(global_ty, new_global))
        }

        ctx.envs.push(globals);

        // We keep capture info separate from the ctx to avoid some borrowing
        // hairballs, since we're generally trying to update the captures just
        // as we get a mutable borrow of a variable in the ctx.
        let mut captures = Vec::new();

        // Time to start analyzing!!
        let mut ast_main = self.ast_main.take().unwrap();
        let (main_idx, _main_ty, _main_captures_ty) =
            self.check_func(&mut ast_main, &mut ctx, &mut cfg, &mut captures);
        cfg.main = main_idx;

        if ctx.envs.len() != 1 {
            self.error(
                format!("Internal Compiler Error: scopes were improperly popped"),
                Span {
                    start: addr(self.input),
                    end: addr(self.input),
                },
            );
        }

        assert!(
            cfg.func_stack.is_empty(),
            "Internal Compiler Error: Funcs were not popped!"
        );

        self.ast_main = Some(ast_main);
        self.cfg = Some(cfg);
        self.ctx = Some(ctx);
    }

    /// Analyze/Compile a function
    fn check_func(
        &mut self,
        func: &mut Function<'p>,
        ctx: &mut TyCtx<'p>,
        cfg: &mut Cfg<'p>,
        captures: &mut Vec<BTreeMap<&'p str, Reg>>,
    ) -> (GlobalFuncIdx, TyIdx, TyIdx) {
        // Give the function's arguments their own scope, and register
        // that scope as the "root" of the function (for the purposes of
        // captures).
        let mut ast_vars = Vec::new();
        let mut ast_var_map = HashMap::new();
        let mut arg_regs = Vec::new();
        let mut arg_tys = Vec::new();
        let mut arg_names = Vec::new();

        // Some of these values are dummies we will fill in later
        let func_idx = cfg.push_global_func(func.name, Vec::new(), ctx.ty_unknown);
        let func_bb = cfg.push_basic_block();

        for decl in &func.args {
            let ty = ctx.memoize_ty(self, &decl.ty);
            if !self.typed || ty != ctx.ty_unknown {
                let reg = cfg.push_reg(ty);
                ast_var_map.insert(decl.ident, ast_vars.len());
                ast_vars.push(Var::reg(ty, reg));
                arg_regs.push(reg);
                arg_tys.push(ty);
                arg_names.push(decl.ident);
            } else {
                self.error(
                    format!("Compile Error: function arguments must have types"),
                    decl.span,
                )
            }
        }

        ctx.envs.push(CheckEnv {
            vars: ast_vars,
            var_map: ast_var_map,
            tys: HashMap::new(),
            block_kind: BlockKind::Func,
        });

        // Start collecting up the captures for this function
        captures.push(BTreeMap::new());

        // Grab the return type, we'll need this to validate return
        // statements in the body.
        let return_ty = if let TyName::Func { return_ty, .. } = &func.ty {
            return_ty
        } else {
            panic!(
                "Internal Compiler Error: function that wasn't a function? {}",
                func.name
            );
        };

        // If the `-> Type` is omitted from a function decl, assume
        // the function returns the empty tuple `()`, just like Rust.
        let mut return_ty = ctx.memoize_ty(self, return_ty);
        if return_ty == ctx.ty_unknown {
            return_ty = ctx.ty_empty;
        }

        let func_ty = ctx.memoize_inner(Ty::Func { arg_tys, return_ty });

        // Fill in dummied values
        {
            let func = cfg.funcs.last_mut().unwrap();
            func.func_ty = func_ty;
            func.func_arg_names = arg_names;
            cfg.bb(func_bb).args = arg_regs;
        }

        // Do the analysis!!
        let func_exit = self.check_block(
            &mut func.stmts,
            ctx,
            cfg,
            func_bb,
            captures,
            return_ty,
            BlockKind::General,
        );
        cfg.bb(func_exit).stmts.push(CfgStmt::ScopeExitForFunc);

        // Cleanup
        func.captures = captures.pop().unwrap();

        let captures_ty = if !func.captures.is_empty() {
            // TODO: properly type the captures
            let arg_tys = func
                .captures
                .iter()
                .map(|(_cap_name, cap)| cap.ty)
                .collect();
            let captures_ty = ctx.memoize_inner(Ty::Tuple(arg_tys));
            let captures_ptr_ty = ctx.memoize_inner(Ty::TypedPtr(captures_ty));
            let captures_arg_reg = cfg.push_reg(captures_ptr_ty);
            cfg.bb(func_bb).args.push(captures_arg_reg);

            let mut prelude = Vec::new();
            for (capture_idx, (_capture_name, capture_temp)) in func.captures.iter().enumerate() {
                prelude.push(CfgStmt::Copy {
                    new_reg: capture_temp.reg,
                    src_var: CfgVarPath::Reg(
                        captures_arg_reg,
                        vec![
                            CfgPathPart::Deref,
                            CfgPathPart::Field(CompositeFieldIdx(capture_idx)),
                        ],
                    ),
                })
            }
            let entry_point = &mut cfg.bb(func_bb).stmts;
            let mut real_entry_point = std::mem::replace(entry_point, prelude);
            entry_point.append(&mut real_entry_point);
            captures_ty
        } else {
            ctx.ty_empty
        };

        ctx.envs.pop();
        assert!(
            cfg.cur_func().loops.is_empty(),
            "Internal Compiler Error: Loops were not popped!"
        );
        cfg.pop_global_func();

        (func_idx, func_ty, captures_ty)
    }

    /// Analyze/Compile a block of the program (fn body, if, loop, ...).
    ///
    /// Returns the exit bb of the scope.
    fn check_block(
        &mut self,
        stmts: &mut [Statement<'p>],
        ctx: &mut TyCtx<'p>,
        cfg: &mut Cfg<'p>,
        mut bb: BasicBlockIdx,
        captures: &mut Vec<BTreeMap<&'p str, Reg>>,
        return_ty: TyIdx,
        block_kind: BlockKind,
    ) -> BasicBlockIdx {
        // Create a new scope for all the local variables declared in this block.
        ctx.envs.push(CheckEnv {
            var_map: HashMap::new(),
            vars: Vec::new(),
            tys: HashMap::new(),
            block_kind,
        });

        let scope_exit = cfg.push_basic_block();
        let scope_exit_discriminant = cfg.push_reg(ctx.ty_int);
        cfg.bb(scope_exit).args.push(scope_exit_discriminant);

        // Now analyze all the statements. We need to:
        //
        // * recursively analyze all sub-expressions (yielding their computed type)
        // * typecheck subexprs (compare their computed type to an expected type, if any)
        // * register variable names and their types (i.e. for `let` and `fn`)
        // * resolve variable names (usually this is done by check_expr, as VarPaths are exprs)
        //   * register captures of those variables (needed by the runtime)
        // * register type decls as new (scoped) types (i.e. struct decls)
        // * check that control flow makes sense (e.g. break/continue validation)
        //
        // A big part of this process is managed by the ctx (TyCtx), which keeps
        // track of all the known types and variables. See `memoize_ty` for details.

        for Statement {
            code: stmt,
            span: stmt_span,
        } in stmts
        {
            match stmt {
                Stmt::If {
                    expr,
                    stmts,
                    else_stmts,
                } => {
                    let expr_temp = self.check_expr(expr, ctx, cfg, bb, captures);
                    let expected_ty = ctx.memoize_ty(self, &TyName::Bool);

                    self.check_ty(ctx, expr_temp.ty, expected_ty, "`if`", expr.span);
                    let if_bb = cfg.push_basic_block();
                    let if_exit = self.check_block(
                        stmts,
                        ctx,
                        cfg,
                        if_bb,
                        captures,
                        return_ty,
                        BlockKind::If,
                    );
                    let else_bb = cfg.push_basic_block();
                    let else_exit = self.check_block(
                        else_stmts,
                        ctx,
                        cfg,
                        else_bb,
                        captures,
                        return_ty,
                        BlockKind::Else,
                    );

                    cfg.bb(bb).stmts.push(CfgStmt::Branch {
                        cond: expr_temp.reg,
                        if_block: BasicBlockJmp {
                            block_id: if_bb,
                            args: Vec::new(),
                        },
                        else_block: BasicBlockJmp {
                            block_id: else_bb,
                            args: Vec::new(),
                        },
                    });

                    let dest_bb = cfg.push_basic_block();

                    let if_exit_discrim = cfg.bb(if_exit).args[0];
                    cfg.bb(if_exit).stmts.push(CfgStmt::ScopeExitForBlock {
                        cond: if_exit_discrim,
                        block_end: BasicBlockJmp {
                            block_id: dest_bb,
                            args: Vec::new(),
                        },
                        parent_scope_exit: BasicBlockJmp {
                            block_id: scope_exit,
                            args: vec![if_exit_discrim],
                        },
                    });

                    let else_exit_discrim = cfg.bb(else_exit).args[0];
                    cfg.bb(else_exit).stmts.push(CfgStmt::ScopeExitForBlock {
                        cond: else_exit_discrim,
                        block_end: BasicBlockJmp {
                            block_id: dest_bb,
                            args: Vec::new(),
                        },
                        parent_scope_exit: BasicBlockJmp {
                            block_id: scope_exit,
                            args: vec![else_exit_discrim],
                        },
                    });

                    bb = dest_bb;
                }
                Stmt::Loop { stmts } => {
                    let loop_bb = cfg.push_basic_block();
                    let post_loop_bb = cfg.push_basic_block();

                    cfg.bb(bb).stmts.push(CfgStmt::Jump(BasicBlockJmp {
                        block_id: loop_bb,
                        args: Vec::new(),
                    }));

                    cfg.push_loop(loop_bb, post_loop_bb);
                    let loop_exit = self.check_block(
                        stmts,
                        ctx,
                        cfg,
                        loop_bb,
                        captures,
                        return_ty,
                        BlockKind::Loop,
                    );

                    // Make the last loop bb jump to the start of the loop
                    let loop_exit_discrim = cfg.bb(loop_exit).args[0];
                    cfg.bb(loop_exit).stmts.push(CfgStmt::ScopeExitForLoop {
                        cond: loop_exit_discrim,
                        loop_start: BasicBlockJmp {
                            block_id: loop_bb,
                            args: Vec::new(),
                        },
                        loop_end: BasicBlockJmp {
                            block_id: post_loop_bb,
                            args: Vec::new(),
                        },
                        parent_scope_exit: BasicBlockJmp {
                            block_id: scope_exit,
                            args: vec![loop_exit_discrim],
                        },
                    });

                    cfg.pop_loop();
                    bb = post_loop_bb;
                }
                Stmt::Struct(struct_decl) => {
                    ctx.push_struct_decl(self, struct_decl.clone());
                }
                Stmt::Func { func } => {
                    // We push a func's name after checking it to avoid
                    // infinite capture recursion. This means naive recursion
                    // is illegal.
                    let (global_func, func_ty, captures_ty) =
                        self.check_func(func, ctx, cfg, captures);

                    let mut capture_regs = Vec::new();
                    for (capture_name, _callee_capture_reg) in &func.captures {
                        let capture_expr = Expression {
                            code: Expr::VarPath(VarPath {
                                ident: capture_name,
                                fields: Vec::new(),
                            }),
                            span: *stmt_span,
                        };
                        let capture_temp = self.check_expr(&capture_expr, ctx, cfg, bb, captures);

                        capture_regs.push(capture_temp.reg);
                    }

                    let new_var = if capture_regs.is_empty() {
                        Var::global_func(func_ty, global_func)
                    } else {
                        let captures_tuple_reg = cfg.push_reg(captures_ty);
                        cfg.bb(bb).stmts.push(CfgStmt::Tuple {
                            new_reg: captures_tuple_reg,
                            args: capture_regs,
                        });
                        let box_ty = ctx.memoize_inner(Ty::TypedPtr(captures_ty));
                        let box_reg = cfg.push_reg(box_ty);
                        cfg.bb(bb)
                            .stmts
                            .push(CfgStmt::HeapAlloc { new_reg: box_reg });
                        cfg.bb(bb).stmts.push(CfgStmt::Set {
                            dest_var: CfgVarPath::Reg(box_reg, vec![CfgPathPart::Deref]),
                            src_reg: captures_tuple_reg,
                        });
                        let new_reg = cfg.push_reg(func_ty);
                        cfg.bb(bb).stmts.push(CfgStmt::Closure {
                            new_reg,
                            global_func,
                            captures_reg: box_reg,
                        });
                        Var::reg(func_ty, new_reg)
                    };

                    ctx.envs.last_mut().unwrap().push_var(func.name, new_var);
                }
                Stmt::Let { name, expr, is_mut } => {
                    let expr_temp = self.check_expr(expr, ctx, cfg, bb, captures);
                    let expected_ty = ctx.memoize_ty(self, &name.ty);

                    // If a let statement has no type annotation, infer it
                    // to have the type of the expr assigned to it. Ultimately
                    // this just means not bothering to type check it.
                    if expected_ty != ctx.ty_unknown {
                        self.check_ty(ctx, expr_temp.ty, expected_ty, "`let`", expr.span);
                    }

                    // Register this variable in the current scope. This may overwrite
                    // an existing variable in the scope, but that's ok because
                    // it has been completely shadowed and can never be referenced again.

                    let new_var = if *is_mut {
                        let ptr_ty = ctx.memoize_inner(Ty::TypedPtr(expr_temp.ty));
                        let alloca_reg = cfg.push_reg(ptr_ty);
                        let stack_alloc = cfg.push_stack_alloc(expr_temp.ty);
                        let new_var = Var::alloca(expr_temp.ty, alloca_reg);
                        cfg.bb(bb).stmts.push(CfgStmt::StackAlloc {
                            new_reg: alloca_reg,
                            alloc: stack_alloc,
                        });

                        cfg.bb(bb).stmts.push(CfgStmt::Set {
                            dest_var: CfgVarPath::Reg(alloca_reg, vec![CfgPathPart::Deref]),
                            src_reg: expr_temp.reg,
                        });

                        new_var
                    } else {
                        Var::reg(expr_temp.ty, expr_temp.reg)
                    };

                    ctx.envs.last_mut().unwrap().push_var(name.ident, new_var);
                }
                Stmt::Set {
                    path: var_path,
                    expr,
                } => {
                    let expr_temp = self.check_expr(expr, ctx, cfg, bb, captures);
                    if let Some(var) = ctx.resolve_var(var_path.ident) {
                        // Only allow locals (non-captures) to be `set`, because captures
                        // are by-value, so setting a capture is probably a mistake we don't
                        // want the user to make.
                        if var.capture_depth != 0 {
                            self.error(
                                format!("Compile Error: Trying to `set` captured variable '{}' (captures are by-value!)", var_path.ident),
                                *stmt_span,
                            )
                        }
                        let var = var.entry.clone();
                        let dest_reg = if let Var::Alloca { reg, .. } = var {
                            reg
                        } else {
                            self.error(
                                format!(
                                    "Compile Error: Trying to `set` immutable var '{}'",
                                    var_path.ident
                                ),
                                *stmt_span,
                            )
                        };
                        let (expected_ty, var_path) = self.resolve_var_path(
                            ctx,
                            var.ty(),
                            &var_path.fields,
                            true,
                            *stmt_span,
                        );
                        self.check_ty(ctx, expr_temp.ty, expected_ty, "`set`", expr.span);

                        cfg.bb(bb).stmts.push(CfgStmt::Set {
                            dest_var: CfgVarPath::Reg(dest_reg, var_path),
                            src_reg: expr_temp.reg,
                        });
                        // Unlike let, we don't actually need to update the variable
                        // because its type isn't allowed to change!
                    } else {
                        self.error(
                            format!(
                                "Compile Error: Trying to `set` undefined variable '{}'",
                                var_path.ident,
                            ),
                            *stmt_span,
                        )
                    }
                }
                Stmt::Ret { expr } => {
                    let expr_temp = self.check_expr(expr, ctx, cfg, bb, captures);

                    self.check_ty(ctx, expr_temp.ty, return_ty, "return", expr.span);

                    // Set the return value
                    cfg.bb(bb).stmts.push(CfgStmt::SetReturn {
                        src_reg: expr_temp.reg,
                    });

                    // "unwind" the function
                    let exit_mode_reg = cfg.push_reg(ctx.ty_int);
                    cfg.bb(bb).stmts.push(CfgStmt::Lit {
                        new_reg: exit_mode_reg,
                        lit: Literal::Int(ScopeExitKind::ExitReturn as i64),
                    });
                    cfg.bb(bb).stmts.push(CfgStmt::Jump(BasicBlockJmp {
                        block_id: scope_exit,
                        args: vec![exit_mode_reg],
                    }));
                }
                Stmt::Print { expr } => {
                    let expr_temp = self.check_expr(expr, ctx, cfg, bb, captures);

                    cfg.bb(bb).stmts.push(CfgStmt::Print {
                        src_reg: expr_temp.reg,
                    });
                    // Print takes any type, it's magic!
                }
                Stmt::Break | Stmt::Continue => {
                    // Only allow these statements if we're inside a loop!
                    let mut found_loop = false;
                    for env in ctx.envs.iter().rev() {
                        match env.block_kind {
                            BlockKind::Loop => {
                                found_loop = true;
                                break;
                            }
                            BlockKind::Func => {
                                // Reached a function boundary, so there's no loop in scope.
                                break;
                            }
                            BlockKind::If | BlockKind::Else | BlockKind::General => {
                                // Do nothing, keep searching
                            }
                        }
                    }

                    if !found_loop {
                        self.error(format!("Compile Error: This isn't in a loop!"), *stmt_span)
                    }

                    let exit_mode = if let Stmt::Break = stmt {
                        ScopeExitKind::ExitBreak
                    } else if let Stmt::Continue = stmt {
                        ScopeExitKind::ExitContinue
                    } else {
                        unreachable!()
                    };
                    let exit_mode_reg = cfg.push_reg(ctx.ty_int);
                    cfg.bb(bb).stmts.push(CfgStmt::Lit {
                        new_reg: exit_mode_reg,
                        lit: Literal::Int(exit_mode as i64),
                    });
                    cfg.bb(bb).stmts.push(CfgStmt::Jump(BasicBlockJmp {
                        block_id: scope_exit,
                        args: vec![exit_mode_reg],
                    }));
                }
            }
        }

        {
            let cur_scope = ctx.envs.last().unwrap();
            for var in cur_scope.vars.iter().rev() {
                match var {
                    Var::Alloca { ty, reg } => {
                        if ctx.needs_drop[*ty] {
                            println!("alloca dropping %{}", reg.0);
                            cfg.bb(scope_exit)
                                .stmts
                                .push(CfgStmt::Drop { target: *reg });
                        }
                        cfg.bb(scope_exit)
                            .stmts
                            .push(CfgStmt::StackDealloc { src_reg: *reg })
                    }
                    Var::Reg { ty, reg } => {
                        if ctx.needs_drop[*ty] {
                            println!("reg dropping %{}", reg.0);
                            cfg.bb(scope_exit)
                                .stmts
                                .push(CfgStmt::Drop { target: *reg });
                        }
                    }
                    Var::GlobalFunc { .. } => {
                        // Do nothing
                    }
                }
            }

            let needs_fallthrough = cfg
                .bb(bb)
                .stmts
                .last()
                .map(|stmt| !matches!(stmt, CfgStmt::Jump(..)))
                .unwrap_or(true);
            if needs_fallthrough {
                let exit_mode_reg = cfg.push_reg(ctx.ty_int);
                cfg.bb(bb).stmts.push(CfgStmt::Lit {
                    new_reg: exit_mode_reg,
                    lit: Literal::Int(ScopeExitKind::ExitNormal as i64),
                });
                cfg.bb(bb).stmts.push(CfgStmt::Jump(BasicBlockJmp {
                    block_id: scope_exit,
                    args: vec![exit_mode_reg],
                }));
            }
        }

        ctx.envs.pop();
        scope_exit
    }

    fn check_expr(
        &mut self,
        expr: &Expression<'p>,
        ctx: &mut TyCtx<'p>,
        cfg: &mut Cfg<'p>,
        bb: BasicBlockIdx,
        captures: &mut Vec<BTreeMap<&'p str, Reg>>,
    ) -> Reg {
        match &expr.code {
            Expr::Lit(lit) => {
                let ty = ctx.memoize_ty(self, &lit.ty());
                let new_reg = cfg.push_reg(ty);
                cfg.bb(bb).stmts.push(CfgStmt::Lit {
                    new_reg,
                    lit: lit.clone(),
                });
                return Reg::new(ty, new_reg);
            }
            Expr::VarPath(var_path) => {
                return self.get_var_path_as_reg(ctx, cfg, var_path, captures, bb, expr.span);
            }
            Expr::Tuple(args) => {
                let mut arg_regs = Vec::new();
                let mut arg_tys = Vec::new();

                for arg in args {
                    let arg_temp = self.check_expr(arg, ctx, cfg, bb, captures);
                    arg_regs.push(arg_temp.reg);
                    arg_tys.push(arg_temp.ty);
                }

                let ty = ctx.memoize_inner(Ty::Tuple(arg_tys));
                let new_reg = cfg.push_reg(ty);
                cfg.bb(bb).stmts.push(CfgStmt::Tuple {
                    new_reg,
                    args: arg_regs,
                });
                return Reg::new(ty, new_reg);
            }
            Expr::Named { name, args } => {
                if let Some(ty_idx) = ctx.resolve_nominal_ty(name) {
                    if let Ty::NamedStruct(ty_decl) = ctx.realize_ty(ty_idx) {
                        let ty_decl = ty_decl.clone();
                        if args.len() != ty_decl.fields.len() {
                            self.error(
                                format!(
                                    "Compile Error: field count mismatch (expected {}, got {})",
                                    ty_decl.fields.len(),
                                    args.len()
                                ),
                                expr.span,
                            )
                        }
                        let mut field_regs = Vec::new();
                        for (field_idx, ((field, arg), field_decl)) in
                            args.iter().zip(ty_decl.fields.iter()).enumerate()
                        {
                            if *field != field_decl.ident {
                                self.error(
                                    format!(
                                        "Compile Error: field name mismatch (expected {}, got {})",
                                        field_decl.ident, field
                                    ),
                                    arg.span,
                                )
                            }

                            let field_temp = self.check_expr(arg, ctx, cfg, bb, captures);
                            let expected_ty = field_decl.ty;
                            self.check_ty(
                                ctx,
                                field_temp.ty,
                                expected_ty,
                                "struct literal",
                                arg.span,
                            );

                            field_regs.push((CompositeFieldIdx(field_idx), field_temp.reg));
                        }

                        let nominal_idx = cfg.push_struct_decl(ty_idx);
                        let new_reg = cfg.push_reg(ty_idx);
                        cfg.bb(bb).stmts.push(CfgStmt::Struct {
                            nominal: nominal_idx,
                            new_reg,
                            fields: field_regs,
                        });

                        Reg::new(ty_idx, new_reg)
                    } else {
                        let ty_str = ctx.format_ty(ty_idx);
                        panic!(
                            "Internal Compiler Error: nominal ty wasn't nominal?? {}: {}",
                            name, ty_str
                        );
                    }
                } else if self.typed {
                    self.error(
                        format!("Compile Error: Use of undefined struct '{}'", name),
                        expr.span,
                    )
                } else {
                    // In untyped mode it's ok for a struct to be used without being declared!
                    for (_field, arg) in args {
                        self.check_expr(arg, ctx, cfg, bb, captures);
                    }
                    let ty = ctx.memoize_ty(self, &TyName::Named(name));
                    let new_reg = cfg.push_reg(ty);
                    Reg::new(ty, new_reg)
                }
            }
            Expr::Call { func, args } => {
                let (func_path, func_ty_idx) =
                    self.get_var_path_as_cfg_path(ctx, cfg, func, captures, bb, expr.span);

                let func_ty = ctx.realize_ty(func_ty_idx).clone();
                let (arg_tys, return_ty) = if let Ty::Func { arg_tys, return_ty } = func_ty {
                    (arg_tys.clone(), return_ty)
                } else if self.typed {
                    self.error(
                        format!("Compile Error: Function call must have Func type!"),
                        expr.span,
                    )
                } else {
                    (Vec::new(), ctx.memoize_ty(self, &TyName::Unknown))
                };

                if self.typed && arg_tys.len() != args.len() {
                    self.error(
                        format!(
                            "Compile Error: arg count mismatch (expected {:?}, got {:?})",
                            arg_tys.len(),
                            args.len(),
                        ),
                        expr.span,
                    )
                }

                let mut arg_regs = Vec::new();
                for (idx, arg) in args.iter().enumerate() {
                    let arg_temp = self.check_expr(arg, ctx, cfg, bb, captures);
                    let expected_ty = arg_tys.get(idx).copied().unwrap_or(ctx.ty_unknown);

                    self.check_ty(ctx, arg_temp.ty, expected_ty, "arg", arg.span);
                    arg_regs.push(arg_temp.reg);
                }
                let new_reg = cfg.push_reg(return_ty);
                cfg.bb(bb).stmts.push(CfgStmt::Call {
                    new_reg,
                    func: func_path,
                    func_ty: func_ty_idx,
                    args: arg_regs,
                });
                return Reg::new(return_ty, new_reg);
            }
        }
    }

    #[track_caller]
    fn check_ty(
        &mut self,
        ctx: &TyCtx<'p>,
        computed_ty: TyIdx,
        expected_ty: TyIdx,
        env_name: &str,
        span: Span,
    ) {
        if self.typed && computed_ty != expected_ty {
            let expected = ctx.format_ty(expected_ty);
            let computed = ctx.format_ty(computed_ty);

            let msg = if expected != computed {
                format!(
                    "Compile Error: {} type mismatch (expected {}, got {})",
                    env_name, expected, computed,
                )
            } else {
                format!(
                    r#"Compile Error: {} type mismatch (expected {}, got {})
NOTE: the types look the same, but the named types have different decls!"#,
                    env_name, expected, computed,
                )
            };
            self.error(msg, span)
        }
    }

    fn resolve_var_path(
        &mut self,
        ctx: &mut TyCtx<'p>,
        root_ty: TyIdx,
        path: &[&'p str],
        needs_deref: bool,
        span: Span,
    ) -> (TyIdx, Vec<CfgPathPart>) {
        if !self.typed {
            return (ctx.ty_unknown, Vec::new());
        }
        let mut cur_ty = root_ty;
        let mut out_path = Vec::new();
        if needs_deref {
            out_path.push(CfgPathPart::Deref);
        }
        'path: for field in path {
            match ctx.realize_ty(cur_ty) {
                Ty::NamedStruct(struct_decl) => {
                    for (field_idx, struct_field) in struct_decl.fields.iter().enumerate() {
                        if &struct_field.ident == field {
                            cur_ty = struct_field.ty;
                            out_path.push(CfgPathPart::Field(CompositeFieldIdx(field_idx)));
                            continue 'path;
                        }
                    }
                    self.error(
                        format!(
                            "Compile Error: {} is not a field of {}",
                            field, struct_decl.name
                        ),
                        span,
                    )
                }
                Ty::Tuple(arg_tys) => {
                    if let Some((field_idx, field_ty)) = field
                        .parse::<usize>()
                        .ok()
                        .and_then(|idx| arg_tys.get(idx).map(|ty| (idx, ty)))
                    {
                        cur_ty = *field_ty;
                        out_path.push(CfgPathPart::Field(CompositeFieldIdx(field_idx)));
                        continue 'path;
                    } else {
                        self.error(
                            format!(
                                "Compile Error: {} is not a field of {}",
                                field,
                                ctx.format_ty(cur_ty)
                            ),
                            span,
                        )
                    }
                }
                _ => self.error(
                    format!(
                        "Compile Error: there are no fields on type {}",
                        ctx.format_ty(cur_ty)
                    ),
                    span,
                ),
            }
        }
        (cur_ty, out_path)
    }

    fn get_var_path_as_reg(
        &mut self,
        ctx: &mut TyCtx<'p>,
        cfg: &mut Cfg<'p>,
        var_path: &VarPath<'p>,
        captures: &mut Vec<BTreeMap<&'p str, Reg>>,
        bb: BasicBlockIdx,
        span: Span,
    ) -> Reg {
        if let Some(var) = ctx.resolve_var(var_path.ident) {
            let capture_depth = var.capture_depth;
            let mut src_var = var.entry.clone();
            let is_global_func = matches!(src_var, Var::GlobalFunc { .. });

            // Don't capture global function pointers
            if !is_global_func {
                for (captures, depth) in captures.iter_mut().rev().zip(0..capture_depth) {
                    let capture_temp = captures.entry(var_path.ident).or_insert_with(|| {
                        let func_idx = cfg.funcs.len() - depth - 1;
                        let func = &mut cfg.funcs[func_idx];
                        let capture_ty = src_var.ty();
                        func.regs.push(RegDecl { ty: capture_ty });
                        let capture_reg = RegIdx(func.regs.len() - 1);
                        Reg {
                            ty: capture_ty,
                            reg: capture_reg,
                        }
                    });

                    if depth == 0 {
                        src_var = Var::reg(capture_temp.ty, capture_temp.reg);
                    }
                }
            }

            if !self.typed {
                return Reg::new(ctx.ty_unknown, RegIdx(0));
            }

            let has_path = !var_path.fields.is_empty();
            let is_alloca = matches!(src_var, Var::Alloca { .. });
            let (final_ty, var_path) =
                self.resolve_var_path(ctx, src_var.ty(), &var_path.fields, is_alloca, span);
            let needs_temp = src_var.needs_temp() || has_path || ctx.needs_drop[final_ty];

            match src_var {
                Var::Alloca { reg, ty } | Var::Reg { reg, ty } => {
                    if needs_temp {
                        let new_reg = cfg.push_reg(final_ty);
                        cfg.bb(bb).stmts.push(CfgStmt::Copy {
                            new_reg,
                            src_var: CfgVarPath::Reg(reg, var_path),
                        });
                        return Reg::new(final_ty, new_reg);
                    } else {
                        return Reg::new(ty, reg);
                    }
                }
                Var::GlobalFunc { global_func, .. } => {
                    let new_reg = cfg.push_reg(final_ty);
                    cfg.bb(bb).stmts.push(CfgStmt::Copy {
                        new_reg,
                        src_var: CfgVarPath::GlobalFunc(global_func),
                    });
                    return Reg::new(final_ty, new_reg);
                }
            }
        } else {
            self.error(
                format!(
                    "Compile Error: Use of undefined variable '{}'",
                    var_path.ident
                ),
                span,
            )
        }
    }

    fn get_var_path_as_cfg_path(
        &mut self,
        ctx: &mut TyCtx<'p>,
        cfg: &mut Cfg<'p>,
        var_path: &VarPath<'p>,
        captures: &mut Vec<BTreeMap<&'p str, Reg>>,
        _bb: BasicBlockIdx,
        span: Span,
    ) -> (CfgVarPath, TyIdx) {
        if let Some(var) = ctx.resolve_var(var_path.ident) {
            let capture_depth = var.capture_depth;
            let mut src_var = var.entry.clone();
            let is_global_func = matches!(src_var, Var::GlobalFunc { .. });

            // Don't capture global function pointers
            if !is_global_func {
                for (captures, depth) in captures.iter_mut().rev().zip(0..capture_depth) {
                    let capture_temp = captures.entry(var_path.ident).or_insert_with(|| {
                        let func_idx = cfg.funcs.len() - depth - 1;
                        let func = &mut cfg.funcs[func_idx];
                        let capture_ty = src_var.ty();
                        func.regs.push(RegDecl { ty: capture_ty });
                        let capture_reg = RegIdx(func.regs.len() - 1);
                        Reg {
                            ty: capture_ty,
                            reg: capture_reg,
                        }
                    });

                    if depth == 0 {
                        src_var = Var::reg(capture_temp.ty, capture_temp.reg);
                    }
                }
            }

            if !self.typed {
                return (CfgVarPath::Reg(RegIdx(0), Vec::new()), ctx.ty_unknown);
            }

            let has_path = !var_path.fields.is_empty();
            let is_alloca = matches!(src_var, Var::Alloca { .. });
            let (final_ty, var_path) =
                self.resolve_var_path(ctx, src_var.ty(), &var_path.fields, is_alloca, span);
            let _needs_temp = src_var.needs_temp() || has_path || ctx.needs_drop[final_ty];

            match src_var {
                Var::Alloca { reg, .. } | Var::Reg { reg, .. } => {
                    (CfgVarPath::Reg(reg, var_path), final_ty)
                }
                Var::GlobalFunc { global_func, .. } => {
                    (CfgVarPath::GlobalFunc(global_func), final_ty)
                }
            }
        } else {
            self.error(
                format!(
                    "Compile Error: Use of undefined variable '{}'",
                    var_path.ident
                ),
                span,
            )
        }
    }
}

/// An SSA CFG IR (Single Static Assignment Control Flow Graph Intermediate Representation).
///
/// This is the preferred format for writing analysis steps after basic type checking
/// and name resolution.
#[derive(Debug)]
pub struct Cfg<'p> {
    /// All the functions, including builtins and main.
    pub funcs: Vec<FuncCfg<'p>>,
    /// All the named types.
    pub nominals: Vec<TyIdx>,
    /// The main function.
    pub main: GlobalFuncIdx,

    // Transient state, only used while building the CFG
    func_stack: Vec<GlobalFuncIdx>,
}

#[derive(Debug)]
pub struct FuncCfg<'p> {
    pub func_name: &'p str,
    pub func_arg_names: Vec<&'p str>,
    pub func_ty: TyIdx,

    pub blocks: Vec<BasicBlock<'p>>,
    pub regs: Vec<RegDecl>,
    pub stack_allocs: Vec<TyIdx>,

    // Transient state, only used while building the CFG
    // (loop_bb, post_loop_bb)
    loops: Vec<(BasicBlockIdx, BasicBlockIdx)>,
}

#[derive(Debug)]
pub struct BasicBlock<'p> {
    pub args: Vec<RegIdx>,
    pub stmts: Vec<CfgStmt<'p>>,
}

#[derive(Debug)]
pub struct BasicBlockJmp {
    pub block_id: BasicBlockIdx,
    pub args: Vec<RegIdx>,
}

#[repr(i64)]
pub enum ScopeExitKind {
    ExitNormal = 0,
    ExitReturn = 1,
    ExitContinue = 2,
    ExitBreak = 3,
}

impl ScopeExitKind {
    pub fn from_int(val: i64) -> Self {
        match val {
            0 => Self::ExitNormal,
            1 => Self::ExitReturn,
            2 => Self::ExitContinue,
            3 => Self::ExitBreak,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
pub enum CfgStmt<'p> {
    Branch {
        cond: RegIdx,
        if_block: BasicBlockJmp,
        else_block: BasicBlockJmp,
    },
    Jump(BasicBlockJmp),
    /// To exit a normal block, you must pass through
    /// this instruction which decides whether to
    /// fallthrough or bubble to the parent scope exit.
    ScopeExitForBlock {
        cond: RegIdx,
        // Where to jump for fallthrough
        block_end: BasicBlockJmp,
        // Where to jump for break/continue/return
        parent_scope_exit: BasicBlockJmp,
    },
    /// To exit a loop, you must pass through
    /// this instruction which decides whether to
    /// break, continue, or bubble to the parent scope exit
    ScopeExitForLoop {
        cond: RegIdx,
        // Where to jump for `continue`
        loop_start: BasicBlockJmp,
        // Where to jump for `break`
        loop_end: BasicBlockJmp,
        // Where to jump for `return`
        parent_scope_exit: BasicBlockJmp,
    },
    /// *truly* return from the function (assumes return value already set)
    ScopeExitForFunc,
    Drop {
        target: RegIdx,
    },
    Copy {
        new_reg: RegIdx,
        src_var: CfgVarPath,
    },
    Lit {
        new_reg: RegIdx,
        lit: Literal<'p>,
    },
    Closure {
        new_reg: RegIdx,
        global_func: GlobalFuncIdx,
        captures_reg: RegIdx,
    },
    Tuple {
        new_reg: RegIdx,
        args: Vec<RegIdx>,
    },
    Struct {
        new_reg: RegIdx,
        nominal: GlobalNominalIdx,
        fields: Vec<(CompositeFieldIdx, RegIdx)>,
    },
    Call {
        new_reg: RegIdx,
        func: CfgVarPath,
        func_ty: TyIdx,
        args: Vec<RegIdx>,
    },
    StackAlloc {
        new_reg: RegIdx,
        alloc: StackAllocIdx,
    },
    HeapAlloc {
        new_reg: RegIdx,
    },
    StackDealloc {
        src_reg: RegIdx,
    },
    HeapDealloc {
        src_reg: RegIdx,
    },
    Set {
        dest_var: CfgVarPath,
        src_reg: RegIdx,
    },
    SetReturn {
        src_reg: RegIdx,
    },
    Print {
        src_reg: RegIdx,
    },
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RegIdx(pub usize);
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct StackAllocIdx(pub usize);
#[derive(Debug, Copy, Clone)]
pub struct BasicBlockIdx(pub usize);
#[derive(Debug, Copy, Clone)]
pub struct GlobalFuncIdx(pub usize);
#[derive(Debug, Copy, Clone)]
pub struct GlobalNominalIdx(pub usize);
#[derive(Debug, Copy, Clone)]
pub struct CompositeFieldIdx(pub usize);

#[derive(Debug, Copy, Clone)]
pub enum CfgPathPart {
    Field(CompositeFieldIdx),
    Deref,
}

#[derive(Debug)]
pub struct RegDecl {
    pub ty: TyIdx,
}
#[derive(Clone, Debug)]
pub enum CfgVarPath {
    Reg(RegIdx, Vec<CfgPathPart>),
    GlobalFunc(GlobalFuncIdx),
}

impl<'p> Cfg<'p> {
    /// Pushes a reg into cur_func, and returns its idx
    fn push_reg(&mut self, ty: TyIdx) -> RegIdx {
        let cur_func = self.cur_func();
        cur_func.regs.push(RegDecl { ty });

        RegIdx(cur_func.regs.len() - 1)
    }
    /// Pushes an alloc into cur_func, and returns its idx
    fn push_stack_alloc(&mut self, ty: TyIdx) -> StackAllocIdx {
        let cur_func = self.cur_func();
        cur_func.stack_allocs.push(ty);

        StackAllocIdx(cur_func.stack_allocs.len() - 1)
    }
    /// Pushes a basic block into cur_func, and returns its idx
    fn push_basic_block(&mut self) -> BasicBlockIdx {
        let cur_func = self.cur_func();
        cur_func.blocks.push(BasicBlock {
            args: Vec::new(),
            stmts: Vec::new(),
        });

        BasicBlockIdx(cur_func.blocks.len() - 1)
    }
    /// Pushes a loop into scope for cur_func
    fn push_loop(&mut self, loop_bb: BasicBlockIdx, post_loop_bb: BasicBlockIdx) {
        self.cur_func().loops.push((loop_bb, post_loop_bb));
    }
    /// Gets the current loop in scope for cur_func
    fn cur_loop(&mut self) -> (BasicBlockIdx, BasicBlockIdx) {
        *self.cur_func().loops.last().unwrap()
    }
    /// Pops a loop from scope for cur_func
    fn pop_loop(&mut self) {
        self.cur_func().loops.pop();
    }
    /// Gets a bb from cur_func
    fn bb(&mut self, bb_idx: BasicBlockIdx) -> &mut BasicBlock<'p> {
        &mut self.cur_func().blocks[bb_idx.0]
    }

    fn push_global_func(
        &mut self,
        func_name: &'p str,
        func_arg_names: Vec<&'p str>,
        func_ty: TyIdx,
    ) -> GlobalFuncIdx {
        self.funcs.push(FuncCfg {
            func_name,
            func_arg_names,
            func_ty,
            stack_allocs: Vec::new(),
            blocks: Vec::new(),
            regs: Vec::new(),
            loops: Vec::new(),
        });

        let idx = GlobalFuncIdx(self.funcs.len() - 1);
        self.func_stack.push(idx);
        idx
    }

    fn cur_func(&mut self) -> &mut FuncCfg<'p> {
        let cur_func = *self.func_stack.last().unwrap();
        &mut self.funcs[cur_func.0]
    }

    fn pop_global_func(&mut self) {
        self.func_stack.pop();
    }

    fn push_struct_decl(&mut self, struct_ty: TyIdx) -> GlobalNominalIdx {
        self.nominals.push(struct_ty);

        GlobalNominalIdx(self.nominals.len() - 1)
    }

    /// Gets a global func
    pub fn func(&self, func_idx: GlobalFuncIdx) -> &FuncCfg<'p> {
        &self.funcs[func_idx.0]
    }
    /// Gets a global nominal
    pub fn nominal(&self, idx: GlobalNominalIdx) -> TyIdx {
        self.nominals[idx.0]
    }

    pub fn format(&self, ctx: &TyCtx<'p>) -> Result<String, std::fmt::Error> {
        let mut f = String::new();
        for (nominal_ty_idx, nominal_ty) in self.nominals.iter().enumerate() {
            if let Ty::NamedStruct(struct_ty) = ctx.realize_ty(*nominal_ty) {
                writeln!(f, "#struct{}_{} {{", nominal_ty_idx, struct_ty.name)?;
                for field in &struct_ty.fields {
                    writeln!(f, "  {}: {}", field.ident, ctx.format_ty(field.ty))?;
                }
                writeln!(f, "}}")?;
                writeln!(f)?;
            } else {
                unreachable!("Internal Compile Error: Unknown nominal type?");
            }
        }

        for (func_idx, func) in self.funcs.iter().enumerate() {
            if func.blocks.is_empty() {
                // No blocks means this is an intrinsics
                write!(f, "[intrinsic] #fn{}_{}(", func_idx, func.func_name)?;
                for (arg_idx, arg) in func.func_arg_names.iter().enumerate() {
                    if arg_idx != 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                writeln!(f, ")")?;
            } else {
                writeln!(f,)?;
                write!(f, "#fn{}_{}(", func_idx, func.func_name)?;
                let arg_regs = func.blocks[0].args.iter();
                let arg_names = func.func_arg_names.iter().chain(Some(&"[closure]"));
                for (arg_idx, (arg_reg_idx, arg_name)) in arg_regs.zip(arg_names).enumerate() {
                    if arg_idx != 0 {
                        write!(f, ", ")?;
                    }
                    let arg_reg = &func.reg(*arg_reg_idx);
                    write!(
                        f,
                        "%{}_{}: {}",
                        arg_reg_idx.0,
                        arg_name,
                        ctx.format_ty(arg_reg.ty)
                    )?;
                }
                writeln!(f, "):")?;
                func.format(&mut f, ctx, self)?;
            }
        }
        Ok(f)
    }
}

impl<'p> FuncCfg<'p> {
    pub fn reg(&self, idx: RegIdx) -> &RegDecl {
        &self.regs[idx.0]
    }
    // pub fn stack_alloc(&self, idx: StackAllocIdx) -> TyIdx {
    //    self.stack_allocs[idx.0]
    // }
    pub fn block(&self, idx: BasicBlockIdx) -> &BasicBlock {
        &self.blocks[idx.0]
    }
    pub fn arg_regs(&self) -> &[RegIdx] {
        &self.blocks[0].args
    }
    fn format<F: std::fmt::Write>(
        &self,
        f: &mut F,
        ctx: &TyCtx<'p>,
        cfg: &Cfg<'p>,
    ) -> std::fmt::Result {
        for (block_id, block) in self.blocks.iter().enumerate() {
            write!(f, "  bb{}(", block_id)?;
            for (arg_idx, arg) in block.args.iter().enumerate() {
                if arg_idx != 0 {
                    write!(f, ", ")?;
                }
                write!(f, "%{}", arg.0)?;
            }
            writeln!(f, "):")?;

            for stmt in &block.stmts {
                match stmt {
                    CfgStmt::Branch {
                        cond,
                        if_block,
                        else_block,
                    } => {
                        write!(f, "    cond %{}: ", cond.0)?;
                        if_block.format(f)?;
                        write!(f, ", ")?;
                        else_block.format(f)?;
                        writeln!(f)?;
                    }
                    CfgStmt::Jump(block) => {
                        write!(f, "    jmp ")?;
                        block.format(f)?;
                        writeln!(f)?;
                    }
                    CfgStmt::Copy { new_reg, src_var } => {
                        match src_var {
                            CfgVarPath::Reg(reg, var_path) => {
                                write!(f, "    %{} = %{}", new_reg.0, reg.0)?;
                                for field in var_path {
                                    write!(f, ".")?;
                                    field.format(f)?;
                                }
                            }
                            CfgVarPath::GlobalFunc(global_func) => {
                                let func = &cfg.func(*global_func);
                                write!(
                                    f,
                                    "    %{} = #fn{}_{}",
                                    new_reg.0, global_func.0, func.func_name
                                )?;
                            }
                        }
                        writeln!(f)?;
                    }
                    CfgStmt::Lit { new_reg, lit } => {
                        writeln!(f, "    %{} = {:?}", new_reg.0, lit)?;
                    }
                    CfgStmt::Closure {
                        new_reg,
                        global_func,
                        captures_reg,
                    } => {
                        let func = &cfg.func(*global_func);
                        writeln!(
                            f,
                            "    %{} = [closure](#fn{}_{}, %{})",
                            new_reg.0, global_func.0, func.func_name, captures_reg.0
                        )?;
                    }
                    CfgStmt::Call {
                        new_reg,
                        func,
                        func_ty: _,
                        args,
                    } => {
                        match func {
                            CfgVarPath::Reg(reg, var_path) => {
                                write!(f, "    %{} = (%{}", new_reg.0, reg.0)?;
                                for field in var_path {
                                    write!(f, ".")?;
                                    field.format(f)?;
                                }
                                write!(f, ")(")?;
                            }
                            CfgVarPath::GlobalFunc(global_func) => {
                                let func = &cfg.func(*global_func);
                                write!(
                                    f,
                                    "    %{} = #fn{}_{}(",
                                    new_reg.0, global_func.0, func.func_name
                                )?;
                            }
                        }
                        for (arg_idx, arg) in args.iter().enumerate() {
                            if arg_idx != 0 {
                                write!(f, ", ")?;
                            }
                            write!(f, "%{}", arg.0)?;
                        }
                        writeln!(f, ")")?;
                    }
                    CfgStmt::StackAlloc { new_reg, alloc } => {
                        writeln!(f, "    %{} = stack_alloc() (slot {})", new_reg.0, alloc.0)?;
                    }
                    CfgStmt::StackDealloc { src_reg } => {
                        writeln!(f, "    stack_dealloc({})", src_reg.0)?;
                    }
                    CfgStmt::HeapAlloc { new_reg } => {
                        writeln!(f, "    %{} = heap_alloc()", new_reg.0)?;
                    }
                    CfgStmt::HeapDealloc { src_reg } => {
                        writeln!(f, "    heap_dealloc({})", src_reg.0)?;
                    }
                    CfgStmt::Set { dest_var, src_reg } => {
                        if let CfgVarPath::Reg(reg, var_path) = dest_var {
                            write!(f, "    %{}", reg.0)?;
                            for field in var_path {
                                write!(f, ".")?;
                                field.format(f)?;
                            }
                            writeln!(f, " = %{}", src_reg.0)?;
                        } else {
                            panic!("Internal Compiler Error: set a global function?");
                        }
                    }
                    CfgStmt::SetReturn { src_reg } => {
                        writeln!(f, "    *ret = %{}", src_reg.0)?;
                    }
                    CfgStmt::Print { src_reg } => {
                        writeln!(f, "    print %{}", src_reg.0)?;
                    }
                    CfgStmt::Tuple { new_reg, args } => {
                        write!(f, "    %{} = (", new_reg.0)?;
                        for (arg_idx, arg) in args.iter().enumerate() {
                            if arg_idx != 0 {
                                write!(f, ", ")?;
                            }
                            write!(f, "%{}", arg.0)?;
                        }
                        if args.len() == 1 {
                            write!(f, ",")?;
                        }
                        writeln!(f, ")")?;
                    }
                    CfgStmt::Struct {
                        nominal,
                        new_reg,
                        fields,
                    } => {
                        let ty_idx = cfg.nominal(*nominal);
                        let ty = ctx.realize_ty(ty_idx);
                        if let Ty::NamedStruct(struct_decl) = ty {
                            write!(
                                f,
                                "    %{} = #struct{}_{}{{ ",
                                new_reg.0, nominal.0, struct_decl.name
                            )?;
                            for (field_idx, (field_name, field_reg)) in fields.iter().enumerate() {
                                if field_idx != 0 {
                                    write!(f, ", ")?;
                                }
                                write!(f, "{}: %{}", field_name.0, field_reg.0)?;
                            }
                            writeln!(f, " }}")?;
                        } else {
                            panic!("Internal Compiler Error: struct wasn't a struct?");
                        }
                    }
                    CfgStmt::ScopeExitForBlock {
                        cond,
                        block_end,
                        parent_scope_exit,
                    } => {
                        write!(f, "    exit? %{}: end(", cond.0)?;
                        block_end.format(f)?;
                        write!(f, "), parent(")?;
                        parent_scope_exit.format(f)?;
                        write!(f, ")")?;
                        writeln!(f)?;
                    }
                    CfgStmt::ScopeExitForLoop {
                        cond,
                        loop_start,
                        loop_end,
                        parent_scope_exit,
                    } => {
                        write!(f, "    loop? %{}: continue(", cond.0)?;
                        loop_start.format(f)?;
                        write!(f, "), break(")?;
                        loop_end.format(f)?;
                        write!(f, "), parent(")?;
                        parent_scope_exit.format(f)?;
                        write!(f, ")")?;
                        writeln!(f)?;
                    }
                    CfgStmt::ScopeExitForFunc => {
                        writeln!(f, "    ret")?;
                    }
                    CfgStmt::Drop { target } => {
                        writeln!(f, "    drop %{}", target.0)?;
                    }
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl BasicBlockJmp {
    fn format<F: std::fmt::Write>(&self, f: &mut F) -> std::fmt::Result {
        write!(f, "bb{}(", self.block_id.0)?;
        for (arg_idx, arg) in self.args.iter().enumerate() {
            if arg_idx != 0 {
                write!(f, ", ")?;
            }
            write!(f, "%{}", arg.0)?;
        }
        write!(f, ")")?;
        Ok(())
    }
}

impl CfgPathPart {
    fn format<F: std::fmt::Write>(&self, f: &mut F) -> std::fmt::Result {
        match self {
            CfgPathPart::Deref => write!(f, "*"),
            CfgPathPart::Field(field) => write!(f, "{}", field.0),
        }
    }
}
