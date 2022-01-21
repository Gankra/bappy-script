//! A simple untyped interpretter of the CFG (IR)
//!
//! Unlike the AST interpretter, this one more blindly assumes
//! the compiler is correct. Too much information has been
//! stripped away to "double check" the compiler's analysis.

use crate::*;

// Hardcode alignment of stack frames to keep things simple.
const FUNC_ALIGN: usize = 8;
const MAX_STACK_SIZE: usize = 1024 * 1024 * 8;
const MAX_HEAP_SIZE: usize = 1024 * 1024 * 16;
const PTR_SIZE: usize = std::mem::size_of::<Ptr>();
const PTR_ALIGN: usize = std::mem::align_of::<Ptr>();

type Ptr = usize;

pub struct CfgInterpretter {
    builtins: Vec<Builtin>,
    ty_layouts: Vec<TyLayout>,
    frame_layouts: Vec<FrameLayout>,

    rodata: Vec<u8>,
    memory: Vec<u8>,
    string_base: usize,
    stack_ptr: usize,
    heap_ptr: usize,
    output: String,
}

#[derive(Clone, Debug)]
struct FuncAbi {
    args_size: usize,
    args_offsets: Vec<usize>,
    return_offset: usize,
}

#[derive(Clone, Debug)]
pub struct FrameLayout {
    /// Stack frame's size.
    pub frame_size: usize,
    /// How much space the caller needs to reserve for this function's args.
    pub args_size: usize,
    /// Offset from the stack pointer to the variable.
    /// This will reach into the caller's stackframe for function args.
    pub reg_offsets: Vec<usize>,
    pub reg_sizes: Vec<usize>,
    pub return_offset: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct FieldLayout {
    size: usize,
    offset: usize,
}
#[derive(Clone, Debug)]
pub struct CompositeLayout {
    size_align: SizeAlign,
    fields: Vec<FieldLayout>,
}
#[derive(Copy, Clone, Debug)]
pub struct SizeAlign {
    size: usize,
    align: usize,
}
#[derive(Clone, Debug)]
pub struct FuncLayout {
    /// Size and alignment of a function pointer (or closure).
    composite: CompositeLayout,
    /// ABI for this function signature -- how much space the caller must reserve
    /// for the arguments of this function.
    abi: FuncAbi,
}

#[derive(Clone, Debug)]
pub enum TyLayout {
    Primitive(SizeAlign),
    Composite(CompositeLayout),
    Func(FuncLayout),
}

impl TyLayout {
    fn size_align(&self) -> SizeAlign {
        match self {
            TyLayout::Primitive(size_align) => *size_align,
            TyLayout::Composite(composite) => composite.size_align,
            TyLayout::Func(func) => func.composite.size_align,
        }
    }
}

impl<'p> Program<'p> {
    /// Run the program!
    pub fn eval_cfg(&mut self) -> i64 {
        let builtins = self.builtins.clone();

        let cfg = self.cfg.take().unwrap();
        let ctx = self.ctx.take().unwrap();

        let main_idx = cfg.main;

        let mut interpretter = CfgInterpretter {
            rodata: self.input.as_bytes().to_owned(),
            string_base: self.input.as_ptr() as usize,
            memory: vec![0; MAX_STACK_SIZE + MAX_HEAP_SIZE],
            stack_ptr: 0,
            heap_ptr: MAX_STACK_SIZE,
            builtins,
            ty_layouts: Vec::new(),
            frame_layouts: Vec::new(),
            output: String::new(),
        };

        interpretter.compute_layouts(&cfg, &ctx);

        // Reserve space for main's return value
        interpretter.stack_ptr += 8;
        // Run main
        interpretter.eval_func(&cfg, &ctx, main_idx);
        assert!(interpretter.stack_ptr == 8);
        // Read main's return value
        self.cfg = Some(cfg);
        self.ctx = Some(ctx);
        self.output = Some(std::mem::replace(&mut interpretter.output, String::new()));
        interpretter.read_int(0)
    }
}

impl CfgInterpretter {
    fn eval_func(&mut self, cfg: &Cfg, ctx: &TyCtx, func_idx: GlobalFuncIdx) {
        let func = &cfg.funcs[func_idx.0];
        if func.blocks.is_empty() {
            return self.eval_builtin(cfg, ctx, func_idx);
        }
        let mut cur_block = 0;

        let flay = self.frame_layouts[func_idx.0].clone();
        self.stack_ptr += flay.frame_size;
        let return_val_ptr = self.stack_ptr - flay.return_offset;

        'eval_loop: loop {
            let block = &func.blocks[cur_block];
            for stmt in &block.stmts {
                match stmt {
                    CfgStmt::Branch {
                        cond,
                        if_block,
                        else_block,
                    } => {
                        let cond_reg = self.reg_ptr(&flay, *cond);
                        if self.read_bool(cond_reg) {
                            cur_block = if_block.block_id.0;
                        } else {
                            cur_block = else_block.block_id.0;
                        }
                        continue 'eval_loop;
                    }
                    CfgStmt::Jump(block) => {
                        cur_block = block.block_id.0;
                        continue 'eval_loop;
                    }
                    CfgStmt::Return { src_reg } => {
                        let size = self.reg_size(&flay, *src_reg);
                        let src_ptr = self.reg_ptr(&flay, *src_reg);
                        self.copy_from_to(src_ptr, return_val_ptr, size);
                        self.stack_ptr -= flay.frame_size;
                        return;
                    }
                    CfgStmt::Call {
                        new_reg,
                        func,
                        args,
                    } => {
                        let size = self.reg_size(&flay, *new_reg);
                        let new_reg_ptr = self.reg_ptr(&flay, *new_reg);

                        let (callee_func_ty, (callee_func_idx, captures_ptr)) = match func {
                            Var::Alloca { ty, reg } | Var::Reg { ty, reg } => {
                                // In the current implementation these are the same
                                let reg_ptr = self.reg_ptr(&flay, *reg);
                                (*ty, self.read_func(reg_ptr))
                            }
                            Var::GlobalFunc { ty, global_func } => (*ty, (*global_func, 0)),
                        };
                        let callee_abi =
                            if let TyLayout::Func(func_layout) = &self.ty_layouts[callee_func_ty] {
                                func_layout.abi.clone()
                            } else {
                                unreachable!("tried to call non-function");
                            };

                        let args_size = callee_abi.args_size;
                        let args_ptr = self.stack_ptr - args_size;
                        for (arg_idx, arg_src_reg) in args.iter().enumerate() {
                            let arg_offset = callee_abi.args_offsets[arg_idx];
                            let arg_dest_ptr = args_ptr + arg_offset;
                            let arg_src_ptr = self.reg_ptr(&flay, *arg_src_reg);
                            let arg_size = self.reg_size(&flay, *arg_src_reg);
                            self.copy_from_to(arg_src_ptr, arg_dest_ptr, arg_size);
                        }
                        {
                            let capture_arg_offset = callee_abi.args_offsets.last().unwrap();
                            let capture_arg_dest_ptr = args_ptr + capture_arg_offset;
                            self.write_ptr(capture_arg_dest_ptr, captures_ptr);
                        }
                        let callee_return_offset = callee_abi.return_offset;
                        let callee_return_val_ptr = args_ptr + callee_return_offset;
                        self.eval_func(cfg, ctx, callee_func_idx);
                        self.copy_from_to(callee_return_val_ptr, new_reg_ptr, size);
                    }
                    CfgStmt::RegFromLit { new_reg, lit } => {
                        let dest_ptr = self.reg_ptr(&flay, *new_reg);
                        match lit {
                            Literal::Int(val) => self.write_int(dest_ptr, *val),
                            Literal::Str(val) => self.write_str(dest_ptr, *val),
                            Literal::Bool(val) => self.write_bool(dest_ptr, *val),
                            Literal::Empty(()) => { /* noop */ }
                        }
                    }
                    CfgStmt::RegFromVarPath {
                        new_reg,
                        src_var,
                        var_path,
                    } => {
                        let dest_ptr = self.reg_ptr(&flay, *new_reg);
                        match src_var {
                            Var::Alloca { ty, reg } | Var::Reg { ty, reg } => {
                                // In the current implementation these are the same
                                if var_path.is_empty() {
                                    let src_ptr = self.reg_ptr(&flay, *reg);
                                    let size = self.reg_size(&flay, *reg);
                                    self.copy_from_to(src_ptr, dest_ptr, size);
                                } else {
                                    let base_ptr = self.reg_ptr(&flay, *reg);
                                    let (src_ptr, size) =
                                        self.resolve_var_path(ctx, base_ptr, *ty, var_path);
                                    self.copy_from_to(src_ptr, dest_ptr, size);
                                }
                            }
                            Var::GlobalFunc { global_func, .. } => {
                                self.write_func(dest_ptr, (*global_func, 0));
                            }
                        }
                    }
                    CfgStmt::RegFromClosure {
                        new_reg,
                        global_func,
                        captures_reg,
                    } => {
                        let dest_ptr = self.reg_ptr(&flay, *new_reg);
                        let captures_ty = func.regs[captures_reg.0].ty;
                        let captures_size_align = self.layout_of(ctx, captures_ty).size_align();
                        let captures_src_ptr = self.reg_ptr(&flay, *captures_reg);
                        let captures_box_ptr = self.heap_alloc(captures_size_align);
                        self.copy_from_to(
                            captures_src_ptr,
                            captures_box_ptr,
                            captures_size_align.size,
                        );
                        self.write_func(dest_ptr, (*global_func, captures_box_ptr));
                    }
                    CfgStmt::Tuple { new_reg, args } => {
                        let tuple_ty = func.regs[new_reg.0].ty;
                        let tuple_layout = self.composite_layout_of(ctx, tuple_ty).clone();
                        let dest_ptr = self.reg_ptr(&flay, *new_reg);

                        for (src_reg, field_layout) in args.iter().zip(tuple_layout.fields.iter()) {
                            let src_ptr = self.reg_ptr(&flay, *src_reg);
                            self.copy_from_to(
                                src_ptr,
                                dest_ptr + field_layout.offset,
                                field_layout.size,
                            );
                        }
                    }
                    CfgStmt::Struct {
                        new_reg, fields, ..
                    } => {
                        let struct_ty = func.regs[new_reg.0].ty;
                        let struct_layout = self.composite_layout_of(ctx, struct_ty).clone();
                        let dest_ptr = self.reg_ptr(&flay, *new_reg);

                        for (src_reg, field_layout) in
                            fields.iter().zip(struct_layout.fields.iter())
                        {
                            let src_ptr = self.reg_ptr(&flay, src_reg.1);
                            self.copy_from_to(
                                src_ptr,
                                dest_ptr + field_layout.offset,
                                field_layout.size,
                            );
                        }
                    }
                    CfgStmt::Alloca { .. } => { /* noop, stack space is preallocated */ }
                    CfgStmt::Set {
                        dest_var,
                        src_reg,
                        var_path,
                    } => {
                        let src_ptr = self.reg_ptr(&flay, *src_reg);
                        match dest_var {
                            Var::Alloca { ty, reg } | Var::Reg { ty, reg } => {
                                // In the current implementation these are the same
                                if var_path.is_empty() {
                                    let dest_ptr = self.reg_ptr(&flay, *reg);
                                    let size = self.reg_size(&flay, *reg);
                                    self.copy_from_to(src_ptr, dest_ptr, size);
                                } else {
                                    let base_ptr = self.reg_ptr(&flay, *reg);
                                    let (dest_ptr, size) =
                                        self.resolve_var_path(ctx, base_ptr, *ty, var_path);
                                    self.copy_from_to(src_ptr, dest_ptr, size);
                                }
                            }
                            Var::GlobalFunc { .. } => {
                                panic!("Tried to set a global function!?");
                            }
                        }
                    }
                    CfgStmt::Print { src_reg } => {
                        let string = self.format_reg(ctx, func, &flay, *src_reg);
                        println!("{}", string);

                        self.output.push_str(&string);
                        self.output.push_str("\n");
                    }
                }
            }
            unreachable!("Basic Block didn't end with control flow?!");
        }
    }

    fn eval_builtin(&mut self, _cfg: &Cfg, _ctx: &TyCtx, func_idx: GlobalFuncIdx) {
        let builtin = &self.builtins[func_idx.0];
        let intrinsic = builtin.cfg_impl;

        intrinsic(self);
    }
    fn copy_from_to(&mut self, src_ptr: Ptr, dest_ptr: Ptr, size: usize) {
        let src_range = src_ptr..src_ptr + size;
        self.memory.copy_within(src_range, dest_ptr);
    }
    fn heap_alloc(&mut self, size_align: SizeAlign) -> Ptr {
        let aligned_ptr = align_val(self.heap_ptr, size_align.align);
        let new_len = aligned_ptr + size_align.size;
        self.heap_ptr = new_len;
        aligned_ptr
    }
    fn heap_dealloc(&mut self, ptr: Ptr) {
        todo!()
    }

    fn reg_ptr(&self, flay: &FrameLayout, reg: RegIdx) -> Ptr {
        self.stack_ptr - flay.reg_offsets[reg.0]
    }
    fn reg_size(&self, flay: &FrameLayout, reg: RegIdx) -> usize {
        flay.reg_sizes[reg.0]
    }

    fn layout_of(&self, ctx: &TyCtx, ty: TyIdx) -> &TyLayout {
        assert!(
            ty != ctx.ty_unknown,
            "CFG interpretter should never encounter Ty::Unknown!"
        );
        &self.ty_layouts[ty]
    }

    fn composite_layout_of(&self, ctx: &TyCtx, ty: TyIdx) -> &CompositeLayout {
        assert!(
            ty != ctx.ty_unknown,
            "CFG interpretter should never encounter Ty::Unknown!"
        );
        if let TyLayout::Composite(composite) = &self.ty_layouts[ty] {
            composite
        } else {
            unreachable!("Composite wasn't a composite?")
        }
    }

    fn resolve_var_path(
        &self,
        ctx: &TyCtx,
        base_ptr: Ptr,
        ty: TyIdx,
        var_path: &[&str],
    ) -> (Ptr, usize) {
        // println!("resolving var path at 0x{:08x}", base_ptr);
        let mut ptr = base_ptr;
        let mut ty = ty;
        let mut size = 0;
        'main: for field_name in var_path {
            // Loop infinitely to handle autoderef
            loop {
                match ctx.realize_ty(ty) {
                    Ty::NamedStruct(struct_ty) => {
                        let layout = self.composite_layout_of(ctx, ty);
                        for (field_idx, field) in struct_ty.fields.iter().enumerate() {
                            if field.ident == *field_name {
                                let field_layout = &layout.fields[field_idx];
                                ptr = ptr + field_layout.offset;
                                size = field_layout.size;
                                ty = field.ty;
                                continue 'main;
                            }
                        }
                        unreachable!("tried to access non-existent field of struct??");
                    }
                    Ty::Tuple(arg_tys) => {
                        let layout = self.composite_layout_of(ctx, ty);
                        let field_idx = field_name
                            .parse::<usize>()
                            .expect("tuple index wasn't an integer?");
                        let field_layout = &layout.fields[field_idx];
                        ptr = ptr + field_layout.offset;
                        size = field_layout.size;
                        ty = arg_tys[field_idx];
                        continue 'main;
                    }
                    Ty::TypedPtr(pointee_ty) => {
                        let new_ptr = self.read_ptr(ptr);
                        // println!("autoderef 0x{:08x} => 0x{:08x} ", ptr, new_ptr);
                        ptr = new_ptr;
                        ty = *pointee_ty;
                    }
                    _ => {
                        unreachable!("tried to access field of non-composite??");
                    }
                }
            }
        }
        // let mut f = String::new();
        // self.format_ptr(&mut f, ctx, ptr, ty, false, 0).unwrap();
        // println!("varpath: {:?} = *0x{:08x} = {}", var_path, ptr, f);

        (ptr, size)
    }

    fn read_bool(&self, src: Ptr) -> bool {
        self.memory[src] == 1
    }
    fn read_func(&self, src: Ptr) -> (GlobalFuncIdx, Ptr) {
        let func = GlobalFuncIdx(self.read_ptr(src));
        let captures = self.read_ptr(src + PTR_SIZE);
        (func, captures)
    }
    fn read_ptr(&self, src: Ptr) -> Ptr {
        self.read_usize(src)
    }
    fn read_usize(&self, src: Ptr) -> usize {
        const SIZE: usize = std::mem::size_of::<usize>();
        let mut buf = [0; SIZE];
        buf.copy_from_slice(&self.memory[src..src + SIZE]);
        usize::from_le_bytes(buf)
    }
    fn read_int(&self, src: Ptr) -> i64 {
        const SIZE: usize = std::mem::size_of::<i64>();
        let mut buf = [0; SIZE];
        buf.copy_from_slice(&self.memory[src..src + SIZE]);
        i64::from_le_bytes(buf)
    }
    fn read_str(&self, src: Ptr) -> &str {
        let offset = self.read_ptr(src);
        let len = self.read_ptr(src + PTR_SIZE);

        std::str::from_utf8(&self.rodata[offset..offset + len]).unwrap()
    }

    fn write_int(&mut self, dest: Ptr, val: i64) {
        let buf = val.to_le_bytes();
        let size = buf.len();
        self.memory[dest..dest + size].copy_from_slice(&buf);
    }
    fn write_bool(&mut self, dest: Ptr, val: bool) {
        self.memory[dest] = if val { 1 } else { 0 };
    }
    fn write_func(&mut self, dest: Ptr, val: (GlobalFuncIdx, Ptr)) {
        self.write_ptr(dest, val.0 .0);
        self.write_ptr(dest + PTR_SIZE, val.1);
    }
    fn write_ptr(&mut self, dest: Ptr, val: Ptr) {
        self.write_usize(dest, val);
    }
    fn write_usize(&mut self, dest: Ptr, val: usize) {
        let buf = val.to_le_bytes();
        let size = buf.len();
        self.memory[dest..dest + size].copy_from_slice(&buf);
    }
    fn write_str(&mut self, dest: Ptr, val: &str) {
        let string_start = val.as_ptr() as usize;
        let offset = string_start - self.string_base;
        self.write_ptr(dest, offset);
        self.write_ptr(dest + PTR_SIZE, val.len());
    }

    fn compute_layouts(&mut self, cfg: &Cfg, ctx: &TyCtx) {
        for ty in &ctx.tys {
            let layout = match ty {
                // Shouldn't be queried, but need to fill in a value to get the indices right
                Ty::Unknown => TyLayout::Primitive(SizeAlign { size: 0, align: 0 }),
                Ty::Int => TyLayout::Primitive(SizeAlign { size: 8, align: 8 }),
                Ty::Str => TyLayout::Primitive(SizeAlign {
                    size: PTR_SIZE * 2,
                    align: PTR_ALIGN,
                }),
                Ty::Bool => TyLayout::Primitive(SizeAlign { size: 1, align: 1 }),
                Ty::Empty => TyLayout::Primitive(SizeAlign { size: 0, align: 1 }),
                Ty::Ptr => TyLayout::Primitive(SizeAlign {
                    size: PTR_SIZE,
                    align: PTR_ALIGN,
                }),
                Ty::TypedPtr(_) => TyLayout::Primitive(SizeAlign {
                    size: PTR_SIZE,
                    align: PTR_ALIGN,
                }),
                Ty::Func { arg_tys, return_ty } => {
                    // Function pointer + closure pointer (may be dummied for non-capturing closures)
                    let size_align = SizeAlign {
                        size: PTR_SIZE * 2,
                        align: PTR_ALIGN,
                    };

                    let mut args_offset = 0;
                    let mut args_align = 1;
                    let mut args_offsets = Vec::new();

                    // Args are pushed in reverse order (so arg[0] is always at the top).
                    // Return value slot is pushed on top of the args. Capture pointer is the
                    // last "real" arg so it's pushed first. This allows us to pass in a capture
                    // pointer even for functions that don't expect a capture.
                    for &arg_ty in Some(&ctx.ty_ptr)
                        .into_iter()
                        .chain(arg_tys.iter().rev())
                        .chain(Some(return_ty))
                    {
                        let arg_layout = self.layout_of(ctx, arg_ty).size_align();
                        let arg_offset = align_val(args_offset, arg_layout.align);

                        args_offset = arg_offset + arg_layout.size;
                        args_align = args_align.max(arg_layout.align);
                        args_offsets.push(arg_offset);
                    }
                    // Don't actually include the return ty in the args, it's confusing.
                    let return_offset = args_offsets.pop().unwrap();
                    // Unreverse things so things are less weird
                    let args_offsets = args_offsets.into_iter().rev().collect();

                    assert!(args_align <= FUNC_ALIGN);
                    // Round to alignment so that the caller.frame_size + callee.args_size is always aligned.
                    let args_size = align_val(args_offset, FUNC_ALIGN);

                    let abi = FuncAbi {
                        args_size,
                        args_offsets,
                        return_offset,
                    };
                    let composite = CompositeLayout {
                        size_align,
                        fields: vec![
                            // func: FuncIdx,
                            FieldLayout {
                                offset: 0,
                                size: PTR_SIZE,
                            },
                            // captures: Box<(T, U, ...)>,
                            FieldLayout {
                                offset: PTR_SIZE,
                                size: PTR_SIZE,
                            },
                        ],
                    };
                    TyLayout::Func(FuncLayout { composite, abi })
                }
                Ty::Tuple(args) => {
                    let mut offset = 0;
                    let mut align = 1;
                    let mut fields = Vec::new();
                    for &arg in args {
                        let arg_layout = self.layout_of(ctx, arg).size_align();
                        let arg_offset = align_val(offset, arg_layout.align);

                        fields.push(FieldLayout {
                            size: arg_layout.size,
                            offset: arg_offset,
                        });

                        offset = arg_offset + arg_layout.size;
                        align = align.max(arg_layout.align);
                    }

                    let size = align_val(offset, align);
                    let layout = SizeAlign { size, align };

                    TyLayout::Composite(CompositeLayout {
                        size_align: layout,
                        fields,
                    })
                }
                Ty::NamedStruct(struct_ty) => {
                    let mut offset = 0;
                    let mut align = 1;
                    let mut fields = Vec::new();
                    for arg in &struct_ty.fields {
                        let arg_layout = self.layout_of(ctx, arg.ty).size_align();
                        let arg_offset = align_val(offset, arg_layout.align);

                        fields.push(FieldLayout {
                            size: arg_layout.size,
                            offset: arg_offset,
                        });

                        offset = arg_offset + arg_layout.size;
                        align = align.max(arg_layout.align);
                    }

                    let size = align_val(offset, align);
                    let layout = SizeAlign { size, align };

                    TyLayout::Composite(CompositeLayout {
                        size_align: layout,
                        fields,
                    })
                }
            };
            self.ty_layouts.push(layout);
        }

        self.frame_layouts = cfg
            .funcs
            .iter()
            .enumerate()
            .map(|(func_idx, func)| -> FrameLayout {
                if func.blocks.is_empty() {
                    return self.builtins[func_idx].layout.clone();
                }
                let func_args = &func.blocks[0].args;
                let func_ty = func.func_ty;

                let mut offset = 0;
                let mut frame_align = 1;
                let mut reg_offsets = Vec::new();
                let mut reg_sizes = Vec::new();

                for (reg_idx, reg) in func.regs.iter().enumerate() {
                    let reg_layout = self.layout_of(ctx, reg.ty).size_align();
                    reg_sizes.push(reg_layout.size);

                    // Function args are part of the callee
                    if !func_args.contains(&RegIdx(reg_idx)) {
                        let reg_offset = align_val(offset, reg_layout.align);

                        reg_offsets.push(reg_offset);

                        offset = reg_offset + reg_layout.size;
                        frame_align = frame_align.max(reg_layout.align);
                    } else {
                        // Stub out the value, will fill in the next pass
                        reg_offsets.push(0);
                    }
                }
                assert!(frame_align <= FUNC_ALIGN);
                // We probably don't need to align this because this isn't the final size.
                // In a second pass we will add some more size for the args of all the
                // functions we can call. This is basically paranoid alignment because
                // I'm too tired to convince myself it can be excluded.
                let frame_size = align_val(offset, FUNC_ALIGN);

                // Values are currently relative to the "frame pointer"
                // but we want them relative to stack pointer
                for offset in &mut reg_offsets {
                    *offset = frame_size - *offset;
                }

                let abi = if let TyLayout::Func(func_layout) = self.layout_of(ctx, func_ty) {
                    &func_layout.abi
                } else {
                    unreachable!();
                };

                let args_size = abi.args_size;
                // Number of args the caller will pass in
                let abi_arg_count = abi.args_offsets.len();
                // Number of args the callee expects
                let actual_arg_count = func_args.len();
                let abi_diff = abi_arg_count - actual_arg_count;

                // To keep function layout uniform, all functions have a "captures" pointer
                // that is unconditionally passed in as the final argument. However functions
                // which don't capture anything aren't *expecting* this argument, so
                // abi_arg_count will be actual_arg_count - 1. This is ok, because our
                // calling convention allows functions to ignore trailing arguments they don't
                // know about.

                if abi_diff != 0 {
                    assert!(abi_diff == 1);
                }
                let return_offset = frame_size + args_size - abi.return_offset;

                // Truncate to the actual arg count to discard unused capture pointers
                for (arg_idx, arg_offset) in abi.args_offsets[..actual_arg_count].iter().enumerate()
                {
                    let arg_reg_idx = func_args[arg_idx];
                    let arg_reg_offset = &mut reg_offsets[arg_reg_idx.0];
                    *arg_reg_offset = frame_size + args_size - arg_offset;
                }

                FrameLayout {
                    frame_size,
                    args_size,
                    reg_offsets,
                    reg_sizes,
                    return_offset,
                }
            })
            .collect();

        // Second pass, add space to the function for each possible call we make
        for (func_idx, func) in cfg.funcs.iter().enumerate() {
            let mut max_arg_size = 0;
            for block in &func.blocks {
                for stmt in &block.stmts {
                    if let CfgStmt::Call { func, .. } = stmt {
                        if let TyLayout::Func(func_layout) = self.layout_of(ctx, func.ty()) {
                            max_arg_size = max_arg_size.max(func_layout.abi.args_size);
                        } else {
                            unreachable!("non-function was called?");
                        }
                    }
                }
            }

            self.frame_layouts[func_idx].frame_size += max_arg_size;
            self.frame_layouts[func_idx].return_offset += max_arg_size;

            for reg_offset in &mut self.frame_layouts[func_idx].reg_offsets {
                *reg_offset += max_arg_size;
            }
        }
    }

    fn format_reg(&self, ctx: &TyCtx, func: &FuncCfg, flay: &FrameLayout, reg: RegIdx) -> String {
        let mut f = String::new();
        let ptr = self.reg_ptr(flay, reg);
        let ty = func.regs[reg.0].ty;
        self.format_ptr(&mut f, ctx, ptr, ty, false, 0).unwrap();
        f
    }

    fn format_ptr<F: std::fmt::Write>(
        &self,
        f: &mut F,
        ctx: &TyCtx,
        ptr: Ptr,
        ty: TyIdx,
        debug: bool,
        indent: usize,
    ) -> std::fmt::Result {
        match ctx.realize_ty(ty) {
            Ty::Int => {
                let val = self.read_int(ptr);
                write!(f, "{}", val)
            }
            Ty::Str => {
                let val = self.read_str(ptr);
                if debug {
                    write!(f, r#""{}""#, val)
                } else {
                    write!(f, "{}", val)
                }
            }
            Ty::Bool => {
                let val = self.read_bool(ptr);
                write!(f, "{}", val)
            }
            Ty::Ptr => {
                let val = self.read_ptr(ptr);
                write!(f, "0x{:08x}", val)
            }
            Ty::TypedPtr(_) => {
                let val = self.read_ptr(ptr);
                write!(f, "0x{:08x}", val)
            }
            Ty::Empty => {
                write!(f, "()")
            }
            Ty::Tuple(arg_tys) => {
                let composite_layout = self.composite_layout_of(ctx, ty);
                write!(f, "(").unwrap();
                for (idx, (arg_ty, arg_layout)) in arg_tys
                    .iter()
                    .zip(composite_layout.fields.iter())
                    .enumerate()
                {
                    if idx != 0 {
                        write!(f, ", ")?;
                    }
                    let arg_ptr = ptr + arg_layout.offset;
                    // Debug print fields unconditionally to make 0 vs "0" clear
                    self.format_ptr(f, ctx, arg_ptr, *arg_ty, true, indent)?;
                }
                write!(f, ")")
            }
            Ty::NamedStruct(struct_ty) => {
                let composite_layout = self.composite_layout_of(ctx, ty);
                write!(f, "{} {{ ", struct_ty.name)?;
                for (idx, (field, field_layout)) in struct_ty
                    .fields
                    .iter()
                    .zip(composite_layout.fields.iter())
                    .enumerate()
                {
                    if idx != 0 {
                        write!(f, ", ")?;
                    }
                    let field_ptr = ptr + field_layout.offset;
                    write!(f, "{}: ", field.ident)?;
                    // Debug print fields unconditionally to make 0 vs "0" clear
                    self.format_ptr(f, ctx, field_ptr, field.ty, true, indent)?;
                }
                write!(f, " }}")
            }
            Ty::Func { arg_tys, return_ty } => {
                /*
                write!(f, "fn {}(", closure.func.name).unwrap();
                for (i, arg) in closure.func.args.iter().enumerate() {
                    if i != 0 {
                        write!(f, ", ").unwrap();
                    }
                    write!(f, "{}", arg.ident).unwrap();
                }
                writeln!(f, ")").unwrap();

                if !closure.captures.is_empty() {
                    let indent = indent + 2;
                    write!(f, "{:indent$}captures:", "", indent = indent).unwrap();
                    for (arg, capture) in &closure.captures {
                        writeln!(f, "").unwrap();
                        let sub_indent = indent + arg.len() + 2;
                        // Debug print captures unconditionally to make 0 vs "0" clear
                        let val = self.format_val(capture, true, sub_indent);
                        write!(f, "{:indent$}- {}: {}", "", arg, val, indent = indent).unwrap();
                    }
                }
                */
                /*
                Val::Builtin(builtin) => {
                    let mut f = String::new();
                    /*
                    write!(f, "builtin {}(", builtin.name).unwrap();
                    for (i, arg) in builtin.args.iter().enumerate() {
                        if i != 0 {
                            write!(f, ", ").unwrap();
                        }
                        write!(f, "{}", arg).unwrap();
                    }
                    writeln!(f, ")").unwrap();
                    f
                    */
                }
                */
                write!(f, "fn <unimplemented print>")
            }
            Ty::Unknown => {
                write!(f, "<unknown>")
            }
        }
    }
}

fn align_val(val: usize, align: usize) -> usize {
    (val + align - 1) / align * align
}

//
//
//
//
//
//
//
//
//
// builtins!
//
//
//
//
//
//
//
//
//
//
//

pub fn cfg_builtin_add(interp: &mut CfgInterpretter) {
    let stack_ptr = interp.stack_ptr;
    let return_val_ptr = stack_ptr - 8;
    let lhs = interp.read_int(stack_ptr - 16);
    let rhs = interp.read_int(stack_ptr - 24);

    let result = lhs + rhs;
    interp.write_int(return_val_ptr, result);
}
pub fn cfg_builtin_sub(interp: &mut CfgInterpretter) {
    let stack_ptr = interp.stack_ptr;
    let return_val_ptr = stack_ptr - 8;
    let lhs = interp.read_int(stack_ptr - 16);
    let rhs = interp.read_int(stack_ptr - 24);

    let result = lhs - rhs;
    interp.write_int(return_val_ptr, result);
}
pub fn cfg_builtin_mul(interp: &mut CfgInterpretter) {
    let stack_ptr = interp.stack_ptr;
    let return_val_ptr = stack_ptr - 8;
    let lhs = interp.read_int(stack_ptr - 16);
    let rhs = interp.read_int(stack_ptr - 24);

    let result = lhs * rhs;
    interp.write_int(return_val_ptr, result);
}
pub fn cfg_builtin_eq(interp: &mut CfgInterpretter) {
    let stack_ptr = interp.stack_ptr;
    let return_val_ptr = stack_ptr - 8;
    let lhs = interp.read_int(stack_ptr - 16);
    let rhs = interp.read_int(stack_ptr - 24);

    let result = lhs == rhs;
    interp.write_bool(return_val_ptr, result);
}
pub fn cfg_builtin_not(interp: &mut CfgInterpretter) {
    let stack_ptr = interp.stack_ptr;
    let return_val_ptr = stack_ptr - 7;
    let lhs = interp.read_bool(stack_ptr - 8);

    let result = !lhs;
    interp.write_bool(return_val_ptr, result);
}
