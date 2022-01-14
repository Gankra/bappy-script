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

use std::collections::{HashMap, HashSet};
use std::fmt::Write;

/// The structure of a type, with all subtypes resolved to type ids (TyIdx).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ty<'p> {
    Int,
    Str,
    Bool,
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
struct TyCtx<'p> {
    /// Whether static types are enabled/enforced.
    is_typed: bool,
    /// The list of every known type.
    ///
    /// These are the "canonical" copies of each type. Types are
    /// registered here via `memoize`, which returns a TyIdx into
    /// this array.
    ///
    /// Types should be compared by checking if they have the same
    /// TyIdx. This allows you to properly compare nominal types
    /// in the face of shadowing and similar situations.
    tys: Vec<Ty<'p>>,

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
    /// The absence of a type annotation, saved for easy comparison.
    ty_unknown: TyIdx,
    /// The empty tuple, saved for easy use.
    ty_empty: TyIdx,
}

/// Information about types for a specific scope.
#[derive(Debug)]
struct CheckEnv<'p> {
    /// The types of variables
    vars: HashMap<&'p str, Var>,
    /// The struct definitions and TyIdx's
    tys: HashMap<&'p str, TyIdx>,

    /// The kind of block this environment was introduced by.
    ///
    /// This is important for analyzing control flow and captures.
    /// e.g. you need to know what scope a `break` or `continue` refers to.
    block_kind: BlockKind,
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
struct Reg {
    ty: TyIdx,
    reg: RegIdx,
}

#[derive(Debug, Clone)]
struct Var {
    ty: TyIdx,
    reg: RegIdx,
    is_alloca: bool,
}

impl Var {
    fn alloca(ty: TyIdx, reg: RegIdx) -> Self {
        Self {
            ty,
            reg,
            is_alloca: true,
        }
    }

    fn reg(ty: TyIdx, reg: RegIdx) -> Self {
        Self {
            ty,
            reg,
            is_alloca: false,
        }
    }
}

impl Reg {
    fn new(ty: TyIdx, reg: RegIdx) -> Self {
        Self { ty, reg }
    }
}

/// The result of a resolve_var lookup, allowing the value (type) to be set.
struct ResolvedVar<'a, 'p> {
    /// How deep the capture was.
    /// 0 = local (no capture)
    /// 1 = captured from parent function
    /// 2 = captured from grandparent function (parent must now capture it too)
    /// ...etc
    capture_depth: usize,
    /// The variable
    entry: std::collections::hash_map::OccupiedEntry<'a, &'p str, Var>,
}

impl<'p> TyCtx<'p> {
    /// Resolve a variable name (its type) at this point in the program.
    fn resolve_var<'a>(&'a mut self, var_name: &'p str) -> Option<ResolvedVar<'a, 'p>> {
        // By default we're accessing locals
        let mut capture_depth = 0;
        use std::collections::hash_map::Entry;
        for env in self.envs.iter_mut().rev() {
            if let Entry::Occupied(entry) = env.vars.entry(var_name) {
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
            .collect();
        let ty_idx = self.tys.len();
        self.tys.push(Ty::NamedStruct(StructTy {
            name: struct_decl.name,
            fields,
        }));
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
                        .collect();
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
            let ty1 = ty.clone();
            let ty2 = ty;
            let idx = self.tys.len();
            self.ty_map.insert(ty1, idx);
            self.tys.push(ty2);
            idx
        }
    }

    /// Get the type-structure (Ty) associated with this type id (TyIdx).
    fn realize_ty(&self, ty: TyIdx) -> &Ty<'p> {
        if self.is_typed {
            self.tys
                .get(ty)
                .expect("Internal Compiler Error: invalid TyIdx")
        } else {
            &Ty::Unknown
        }
    }

    /// Stringify a type.
    fn format_ty(&self, ty: TyIdx) -> String {
        match self.realize_ty(ty) {
            Ty::Int => format!("Int"),
            Ty::Str => format!("Str"),
            Ty::Bool => format!("Bool"),
            Ty::Empty => format!("()"),
            Ty::Unknown => format!("<unknown>"),
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
            ty_map: HashMap::new(),
            envs: Vec::new(),
            is_typed: self.typed,
            ty_unknown: 0,
            ty_empty: 0,
        };

        // Cache some key types
        ctx.ty_unknown = ctx.memoize_inner(Ty::Unknown);
        ctx.ty_empty = ctx.memoize_inner(Ty::Empty);

        // Set up globals (stdlib)
        let builtins = self
            .builtins
            .clone()
            .iter()
            .map(|builtin| {
                (
                    builtin.name,
                    Var::reg(ctx.memoize_ty(self, &builtin.ty), usize::MAX),
                )
            }) // TODO
            .collect();
        let globals = CheckEnv {
            vars: builtins,
            tys: HashMap::new(),
            // Doesn't really matter what this value is for the globals
            block_kind: BlockKind::General,
        };
        ctx.envs.push(globals);

        // We keep capture info separate from the ctx to avoid some borrowing
        // hairballs, since we're generally trying to update the captures just
        // as we get a mutable borrow of a variable in the ctx.
        let mut captures = Vec::new();

        // Time to start analyzing!!
        let mut main = self.main.take().unwrap();
        self.check_func(&mut main, &mut ctx, &mut captures);

        if ctx.envs.len() != 1 {
            self.error(
                format!("Internal Compiler Error: scopes were improperly popped"),
                Span {
                    start: addr(self.input),
                    end: addr(self.input),
                },
            );
        }

        self.main = Some(main);
    }

    /// Analyze/Compile a function
    fn check_func(
        &mut self,
        func: &mut Function<'p>,
        ctx: &mut TyCtx<'p>,
        captures: &mut Vec<HashSet<&'p str>>,
    ) -> Cfg<'p> {
        let mut cfg = Cfg {
            blocks: Vec::new(),
            regs: Vec::new(),
            loops: Vec::new(),
        };

        // Give the function's arguments their own scope, and register
        // that scope as the "root" of the function (for the purposes of
        // captures).
        let mut ast_vars = HashMap::new();
        let mut cfg_args = Vec::new();

        for decl in &func.args {
            let ty = ctx.memoize_ty(self, &decl.ty);
            if !self.typed || ty != ctx.ty_unknown {
                let reg = cfg.push_reg(ty);
                ast_vars.insert(decl.ident, Var::alloca(ty, reg));
                cfg_args.push(reg);
            } else {
                self.error(
                    format!("Compile Error: function arguments must have types"),
                    decl.span,
                )
            }
        }

        let func_bb = cfg.push_basic_block();
        cfg.bb(func_bb).args = cfg_args;

        ctx.envs.push(CheckEnv {
            vars: ast_vars,
            tys: HashMap::new(),
            block_kind: BlockKind::Func,
        });

        // Start collecting up the captures for this function
        captures.push(HashSet::new());

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

        // Do the analysis!!
        self.check_block(
            &mut func.stmts,
            ctx,
            &mut cfg,
            func_bb,
            captures,
            return_ty,
            BlockKind::General,
        );

        // Cleanup
        func.captures = captures.pop().unwrap();
        ctx.envs.pop();

        cfg.print();
        cfg
    }

    /// Analyze/Compile a block of the program (fn body, if, loop, ...).
    fn check_block(
        &mut self,
        stmts: &mut [Statement<'p>],
        ctx: &mut TyCtx<'p>,
        cfg: &mut Cfg<'p>,
        mut bb: BasicBlockIdx,
        captures: &mut Vec<HashSet<&'p str>>,
        return_ty: TyIdx,
        block_kind: BlockKind,
    ) {
        // Create a new scope for all the local variables declared in this block.
        ctx.envs.push(CheckEnv {
            vars: HashMap::new(),
            tys: HashMap::new(),
            block_kind,
        });

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
                    self.check_block(stmts, ctx, cfg, if_bb, captures, return_ty, BlockKind::If);
                    let else_bb = cfg.push_basic_block();
                    self.check_block(
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

                    cfg.bb(if_bb).stmts.push(CfgStmt::Jump(BasicBlockJmp {
                        block_id: dest_bb,
                        args: Vec::new(),
                    }));

                    cfg.bb(else_bb).stmts.push(CfgStmt::Jump(BasicBlockJmp {
                        block_id: dest_bb,
                        args: Vec::new(),
                    }));

                    bb = dest_bb;
                }
                Stmt::Loop { stmts } => {
                    let loop_bb = cfg.push_basic_block();
                    let post_loop_bb = cfg.push_basic_block();

                    cfg.push_loop(loop_bb, post_loop_bb);
                    self.check_block(
                        stmts,
                        ctx,
                        cfg,
                        loop_bb,
                        captures,
                        return_ty,
                        BlockKind::Loop,
                    );

                    cfg.bb(bb).stmts.push(CfgStmt::Jump(BasicBlockJmp {
                        block_id: loop_bb,
                        args: Vec::new(),
                    }));

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
                    self.check_func(func, ctx, captures);

                    let func_ty = ctx.memoize_ty(self, &func.ty);

                    let captures = Vec::new(); // TODO!!!

                    let new_reg = cfg.push_reg(func_ty);
                    cfg.bb(bb)
                        .stmts
                        .push(CfgStmt::RegFromFunc { new_reg, captures });

                    let func_var = Var::reg(func_ty, new_reg);
                    ctx.envs
                        .last_mut()
                        .unwrap()
                        .vars
                        .insert(func.name, func_var);
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
                        let alloca_reg = cfg.push_reg(expr_temp.ty);
                        let new_var = Var::alloca(expr_temp.ty, alloca_reg);
                        cfg.bb(bb).stmts.push(CfgStmt::Alloca {
                            new_reg: alloca_reg,
                        });

                        cfg.bb(bb).stmts.push(CfgStmt::Set {
                            dest_var: new_var.clone(),
                            var_path: Vec::new(),
                            src_reg: expr_temp.reg,
                        });

                        new_var
                    } else {
                        Var::reg(expr_temp.ty, expr_temp.reg)
                    };

                    ctx.envs
                        .last_mut()
                        .unwrap()
                        .vars
                        .insert(name.ident, new_var);
                }
                Stmt::Set {
                    path: var_path,
                    expr,
                } => {
                    let expr_reg = self.check_expr(expr, ctx, cfg, bb, captures);
                    if let Some(var) = ctx.resolve_var(var_path.ident) {
                        // Only allow locals (non-captures) to be `set`, because captures
                        // are by-value, so setting a capture is probably a mistake we don't
                        // want the user to make.
                        if var.capture_depth == 0 {
                            let var = var.entry.get().clone();
                            if !var.is_alloca {
                                self.error(
                                    format!(
                                        "Compile Error: Trying to `set` immutable var '{}'",
                                        var_path.ident
                                    ),
                                    *stmt_span,
                                )
                            }
                            let expected_ty =
                                self.resolve_var_path(ctx, var.ty, &var_path.fields, *stmt_span);
                            self.check_ty(ctx, expr_reg.ty, expected_ty, "`set`", expr.span);

                            cfg.bb(bb).stmts.push(CfgStmt::Set {
                                dest_var: var,
                                src_reg: expr_reg.reg,
                                var_path: var_path.fields.clone(),
                            });
                            // Unlike let, we don't actually need to update the variable
                            // because its type isn't allowed to change!
                        } else {
                            self.error(
                                format!("Compile Error: Trying to `set` captured variable '{}' (captures are by-value!)", var_path.ident),
                                *stmt_span,
                            )
                        }
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

                    cfg.bb(bb).stmts.push(CfgStmt::Return {
                        src_reg: expr_temp.reg,
                    });
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

                    let (loop_bb, post_loop_bb) = *cfg.loops.last().unwrap();

                    if let Stmt::Break = stmt {
                        cfg.bb(bb).stmts.push(CfgStmt::Jump(BasicBlockJmp {
                            block_id: post_loop_bb,
                            args: Vec::new(),
                        }));
                    } else if let Stmt::Continue = stmt {
                        cfg.bb(bb).stmts.push(CfgStmt::Jump(BasicBlockJmp {
                            block_id: loop_bb,
                            args: Vec::new(),
                        }));
                    } else {
                        unreachable!()
                    }
                }
            }
        }
        ctx.envs.pop();
    }

    fn check_expr(
        &mut self,
        expr: &Expression<'p>,
        ctx: &mut TyCtx<'p>,
        cfg: &mut Cfg<'p>,
        bb: BasicBlockIdx,
        captures: &mut Vec<HashSet<&'p str>>,
    ) -> Reg {
        match &expr.code {
            Expr::Lit(lit) => {
                let ty = ctx.memoize_ty(self, &lit.ty());
                let new_reg = cfg.push_reg(ty);
                cfg.bb(bb).stmts.push(CfgStmt::RegFromLit {
                    new_reg,
                    lit: lit.clone(),
                });
                return Reg::new(ty, new_reg);
            }
            Expr::VarPath(var_path) => {
                if let Some(var) = ctx.resolve_var(var_path.ident) {
                    for (captures, _) in captures.iter_mut().rev().zip(0..var.capture_depth) {
                        captures.insert(var_path.ident);
                    }
                    let src_var = var.entry.get().clone();
                    let ty = self.resolve_var_path(ctx, src_var.ty, &var_path.fields, expr.span);

                    if src_var.is_alloca || !var_path.fields.is_empty() {
                        let new_reg = cfg.push_reg(ty);
                        cfg.bb(bb).stmts.push(CfgStmt::RegFromVarPath {
                            new_reg,
                            src_var,
                            var_path: var_path.fields.clone(),
                        });
                        return Reg::new(ty, new_reg);
                    } else {
                        return Reg::new(src_var.ty, src_var.reg);
                    }
                } else {
                    self.error(
                        format!(
                            "Compile Error: Use of undefined variable '{}'",
                            var_path.ident
                        ),
                        expr.span,
                    )
                }
            }
            Expr::Tuple(args) => {
                let mut arg_regs = Vec::new();
                let mut arg_tys = Vec::new();

                for arg in args {
                    let reg = self.check_expr(arg, ctx, cfg, bb, captures);
                    arg_regs.push(reg.reg);
                    arg_tys.push(reg.ty);
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
                        for ((field, arg), field_decl) in args.iter().zip(ty_decl.fields.iter()) {
                            if *field != field_decl.ident {
                                self.error(
                                    format!(
                                        "Compile Error: field name mismatch (expected {}, got {})",
                                        field_decl.ident, field
                                    ),
                                    arg.span,
                                )
                            }

                            let expr_reg = self.check_expr(arg, ctx, cfg, bb, captures);
                            let expected_ty = field_decl.ty;
                            self.check_ty(
                                ctx,
                                expr_reg.ty,
                                expected_ty,
                                "struct literal",
                                arg.span,
                            );

                            let field_reg = cfg.push_reg(expr_reg.ty);
                            field_regs.push((field_decl.ident, field_reg));
                        }

                        let new_reg = cfg.push_reg(ty_idx);
                        cfg.bb(bb).stmts.push(CfgStmt::Struct {
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
                if let Some(var) = ctx.resolve_var(func) {
                    for (captures, _) in captures.iter_mut().rev().zip(0..var.capture_depth) {
                        captures.insert(func);
                    }

                    let func_var = var.entry.get().clone();
                    let func_ty = ctx.realize_ty(func_var.ty).clone();
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
                        let expr_reg = self.check_expr(arg, ctx, cfg, bb, captures);
                        let expected_ty = arg_tys
                            .get(idx)
                            .copied()
                            .unwrap_or(ctx.memoize_ty(self, &TyName::Unknown));

                        self.check_ty(ctx, expr_reg.ty, expected_ty, "arg", arg.span);
                        arg_regs.push(expr_reg.reg);
                    }
                    let new_reg = cfg.push_reg(return_ty);
                    cfg.bb(bb).stmts.push(CfgStmt::Call {
                        new_reg,
                        func: func_var,
                        args: arg_regs,
                    });
                    return Reg::new(return_ty, new_reg);
                } else {
                    self.error(
                        format!("Compile Error: Call of undefined function '{}'", func),
                        expr.span,
                    )
                }
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
        span: Span,
    ) -> TyIdx {
        let mut cur_ty = root_ty;
        'path: for field in path {
            match ctx.realize_ty(cur_ty) {
                Ty::NamedStruct(struct_decl) => {
                    for struct_field in &struct_decl.fields {
                        if &struct_field.ident == field {
                            cur_ty = struct_field.ty;
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
                    if let Some(field_ty) =
                        field.parse::<usize>().ok().and_then(|idx| arg_tys.get(idx))
                    {
                        cur_ty = *field_ty;
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
        cur_ty
    }
}

struct Cfg<'p> {
    // (loop_bb, post_loop_bb)
    loops: Vec<(BasicBlockIdx, BasicBlockIdx)>,
    blocks: Vec<BasicBlock<'p>>,
    regs: Vec<RegDecl>,
}

struct BasicBlock<'p> {
    args: Vec<RegIdx>,
    stmts: Vec<CfgStmt<'p>>,
}

struct BasicBlockJmp {
    block_id: BasicBlockIdx,
    args: Vec<RegIdx>,
}
enum CfgStmt<'p> {
    Branch {
        cond: RegIdx,
        if_block: BasicBlockJmp,
        else_block: BasicBlockJmp,
    },
    Jump(BasicBlockJmp),
    RegFromVarPath {
        new_reg: RegIdx,
        src_var: Var,
        var_path: Vec<&'p str>,
    },
    RegFromLit {
        new_reg: RegIdx,
        lit: Literal<'p>,
    },
    RegFromFunc {
        new_reg: RegIdx,
        captures: Vec<RegIdx>,
    },
    Tuple {
        new_reg: RegIdx,
        args: Vec<RegIdx>,
    },
    Struct {
        new_reg: RegIdx,
        fields: Vec<(&'p str, RegIdx)>,
    },
    Call {
        new_reg: RegIdx,
        func: Var,
        args: Vec<RegIdx>,
    },
    Alloca {
        new_reg: RegIdx,
    },
    Set {
        dest_var: Var,
        src_reg: RegIdx,
        var_path: Vec<&'p str>,
    },
    Return {
        src_reg: RegIdx,
    },
    Print {
        src_reg: RegIdx,
    },
}

type RegIdx = usize;
type BasicBlockIdx = usize;

struct RegDecl {
    ty: TyIdx,
}

impl<'p> Cfg<'p> {
    fn push_reg(&mut self, ty: TyIdx) -> RegIdx {
        self.regs.push(RegDecl { ty });
        self.regs.len() - 1
    }

    fn push_basic_block(&mut self) -> BasicBlockIdx {
        self.blocks.push(BasicBlock {
            args: Vec::new(),
            stmts: Vec::new(),
        });

        self.blocks.len() - 1
    }

    fn push_loop(&mut self, loop_bb: BasicBlockIdx, post_loop_bb: BasicBlockIdx) {
        self.loops.push((loop_bb, post_loop_bb));
    }

    fn pop_loop(&mut self) {
        self.loops.pop();
    }

    fn bb(&mut self, bb_idx: BasicBlockIdx) -> &mut BasicBlock<'p> {
        &mut self.blocks[bb_idx]
    }

    fn print(&self) {
        for (block_id, block) in self.blocks.iter().enumerate() {
            print!("bb{}(", block_id);
            for (arg_idx, arg) in block.args.iter().enumerate() {
                if arg_idx != 0 {
                    print!(", ");
                }
                print!("%{}", arg);
            }
            println!("):");

            for stmt in &block.stmts {
                match stmt {
                    CfgStmt::Branch {
                        cond,
                        if_block,
                        else_block,
                    } => {
                        print!("  cond %{}: ", cond);
                        if_block.print();
                        print!(", ");
                        else_block.print();
                        println!();
                    }
                    CfgStmt::Jump(block) => {
                        print!("  jmp ");
                        block.print();
                        println!();
                    }
                    CfgStmt::RegFromVarPath {
                        new_reg,
                        src_var,
                        var_path,
                    } => {
                        if src_var.is_alloca {
                            print!("  %{} = (*%{})", new_reg, src_var.reg);
                        } else {
                            print!("  %{} = (%{})", new_reg, src_var.reg);
                        }
                        for field in var_path {
                            print!(".{}", field);
                        }
                        println!();
                    }
                    CfgStmt::RegFromLit { new_reg, lit } => {
                        println!("  %{} = {:?}", new_reg, lit);
                    }
                    CfgStmt::RegFromFunc { new_reg, captures } => {
                        // TODO
                        println!("  %{} = fn(...)", new_reg);
                    }
                    CfgStmt::Call {
                        new_reg,
                        func,
                        args,
                    } => {
                        print!("  %{} = (*%{})(", new_reg, func.reg);
                        for (arg_idx, arg) in args.iter().enumerate() {
                            if arg_idx != 0 {
                                print!(", ");
                            }
                            print!("%{}", arg);
                        }
                        println!(")");
                    }
                    CfgStmt::Alloca { new_reg } => {
                        println!("  %{} = alloca()", new_reg);
                    }
                    CfgStmt::Set {
                        dest_var,
                        src_reg,
                        var_path,
                    } => {
                        print!("  (*%{})", dest_var.reg);
                        for field in var_path {
                            print!(".{}", field);
                        }
                        println!(" = %{}", src_reg);
                    }
                    CfgStmt::Return { src_reg } => {
                        println!("  ret %{}", src_reg);
                    }
                    CfgStmt::Print { src_reg } => {
                        println!("  print %{}", src_reg);
                    }
                    CfgStmt::Tuple { new_reg, args } => {
                        print!("  %{} = (", new_reg);
                        for (arg_idx, arg) in args.iter().enumerate() {
                            if arg_idx != 0 {
                                print!(", ");
                            }
                            print!("%{}", arg);
                        }
                        println!(")");
                    }
                    CfgStmt::Struct { new_reg, fields } => {
                        print!("  %{} = {{", new_reg);
                        for (field_idx, (field_name, field_reg)) in fields.iter().enumerate() {
                            if field_idx != 0 {
                                print!(", ");
                            }
                            print!("{}: %{}", field_name, field_reg);
                        }
                        println!("}}");
                    }
                }
            }
            println!();
        }
    }
}

impl BasicBlockJmp {
    fn print(&self) {
        print!("bb{}(", self.block_id);
        for (arg_idx, arg) in self.args.iter().enumerate() {
            if arg_idx != 0 {
                print!(", ");
            }
            print!("%{}", arg);
        }
        print!(")");
    }
}
