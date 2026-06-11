const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const model = @import("model.zig");

const log = std.log.scoped(.qwen3_vl);

pub const CompilationParameters = struct {
    // Vision exe shapes
    patches: zml.Tensor,            // [max_patches, pd]
    h_pos: zml.Tensor,              // [max_patches] i64
    w_pos: zml.Tensor,              // [max_patches] i64
    attn_mask_v: zml.Tensor,        // [max_patches] f32
    visual_mask: zml.Tensor,        // [max_n_visual] f32
    grid_h: zml.Tensor,             // [] i64
    grid_w: zml.Tensor,             // [] i64

    // Prefill exe shapes
    prefill_tokens: zml.Tensor,     // [max_seqlen] u32
    position_ids: zml.Tensor,       // [3, max_seqlen] i64
    visual_embeds: zml.Tensor,      // [max_n_visual, hidden] bf16
    deepstack: [3]zml.Tensor,       // [max_n_visual, hidden] bf16
    visual_scatter_idx: zml.Tensor, // [max_n_visual] i32
    last_real_pos: zml.Tensor,      // [] u32

    // Decode exe shapes
    decode_token: zml.Tensor,       // [1] u32
    decode_position_ids: zml.Tensor,// [3, 1] i64

    // Shared shapes
    token_index: zml.Tensor,        // [] u32
    kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,

    max_seqlen: u32,
    max_patches: u32,
    max_n_visual: u32,
    shardings: common.Shardings,

    pub fn init(
        mdl: model.Model,
        config: model.Config,
        max_seqlen: u32,
        max_patches: u32,
        shardings: common.Shardings,
    ) CompilationParameters {
        const dtype = mdl.text_model.embed_tokens.weight.dtype();
        const hidden = config.text_config.hidden_size;
        const merge: u32 = @intCast(config.vision_config.spatial_merge_size);
        const max_n_visual: u32 = @divExact(max_patches, merge * merge);
        const pd = config.vision_config.patchEmbedDim();

        return .{
            .patches = .init(.{ .p = max_patches, .pd = pd }, .f32),
            .h_pos = .init(.{ .p = max_patches }, .i64),
            .w_pos = .init(.{ .p = max_patches }, .i64),
            .attn_mask_v = .init(.{ .p = max_patches }, .f32),
            .visual_mask = .init(.{ .n = max_n_visual }, .f32),
            .grid_h = .init(.{}, .i64),
            .grid_w = .init(.{}, .i64),

            .prefill_tokens = .init(.{ .s = max_seqlen }, .u32),
            .position_ids = .init(.{ .mrope = 3, .s = max_seqlen }, .i64),
            .visual_embeds = .init(.{ .n = max_n_visual, .d = hidden }, dtype),
            .deepstack = .{ .init(.{ .n = max_n_visual, .d = hidden }, dtype),
                            .init(.{ .n = max_n_visual, .d = hidden }, dtype),
                            .init(.{ .n = max_n_visual, .d = hidden }, dtype) },
            .visual_scatter_idx = .init(.{ .n = max_n_visual }, .i32),
            .last_real_pos = .init(.{}, .u32),

            .decode_token = .init(.{ .s = 1 }, .u32),
            .decode_position_ids = .init(.{ .mrope = 3, .s = 1 }, .i64),

            .token_index = .init(.{}, .u32),
            .kv_cache = .init(config.text_config, max_seqlen, dtype),
            .rng = .init(),

            .max_seqlen = max_seqlen,
            .max_patches = max_patches,
            .max_n_visual = max_n_visual,
            .shardings = shardings,
        };
    }
};

pub const CompiledModel = struct {
    loaded_model: *const model.LoadedModel,
    vision_exe: zml.Exe,
    prefill_exe: zml.Exe,
    decode_exe: zml.Exe,
    params: CompilationParameters,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        loaded_model: *const model.LoadedModel,
        qwen_model: model.Model,
        parameters: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        return .{
            .loaded_model = loaded_model,
            .vision_exe = try compileVision(allocator, io, platform, qwen_model, parameters, progress),
            .prefill_exe = try compilePrefill(allocator, io, platform, qwen_model, parameters, progress),
            .decode_exe = try compileDecode(allocator, io, platform, qwen_model, parameters, progress),
            .params = parameters,
        };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.vision_exe.deinit();
        self.prefill_exe.deinit();
        self.decode_exe.deinit();
    }
};

fn compileVision(allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    qwen_model: model.Model,
    params: CompilationParameters,
    progress: *std.Progress.Node,
) !zml.Exe {

    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling vision...", 1);
    defer node.end();

    const from: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled vision [{f}]", .{from.untilNow(io, .awake)});

    return platform.compile(
        allocator, io,
        qwen_model.vision_model,
        .paddedVisionForward,
        .{
            params.patches,
            params.h_pos,
            params.w_pos,
            params.attn_mask_v,
            params.visual_mask,
            qwen_model.config.vision_config.deepstack_visual_indexes,
            params.grid_h,
            params.grid_w,
        },
        .{
            .shardings = &params.shardings.all(),
            .program_name = "llm_qwen3_vl_vision",
        },
    );
}

fn compilePrefill(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    qwen_model: model.Model,
    params: CompilationParameters,
    progress: *std.Progress.Node,
) !zml.Exe {

    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling prefill...", 1);
    defer node.end();

    const from: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled prefill [{f}]", .{from.untilNow(io, .awake)});

    return platform.compile(
        allocator, io,
        qwen_model,
        .prefill,
        .{
            params.prefill_tokens,
            params.position_ids,
            params.visual_embeds,
            params.deepstack,
            params.visual_scatter_idx,
            params.token_index,
            params.last_real_pos,
            params.kv_cache,
            params.rng,
        },
        .{
            .shardings = &params.shardings.all(),
            .program_name = "llm_qwen3_vl_prefill",
        },
    );
}


fn compileDecode(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    qwen_model: model.Model,
    params: CompilationParameters,
    progress: *std.Progress.Node,
) !zml.Exe {

    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling decode...", 1);
    defer node.end();

    const from: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled decode [{f}]", .{from.untilNow(io, .awake)});

    return platform.compile(
        allocator, io,
        qwen_model,
        .forward,
        .{
            params.decode_token,
            params.decode_position_ids,
            params.token_index,
            params.kv_cache,
            params.rng,
        },
        .{
            .shardings = &params.shardings.all(),
            .program_name = "llm_qwen3_vl_decode",
        },
    );
}
