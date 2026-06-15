const std = @import("std");

const zml = @import("zml");

const inference = @import("inference.zig");
const model = @import("model.zig");
const image = @import("image.zig");

const log = std.log.scoped(.qwen3_vl);

pub const PreprocessedImage = image.Preprocessed;

pub const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_buffers: *model.Buffers,
    compiled_model: *const inference.CompiledModel,

    kv_cache_buffers: zml.Bufferized(model.KvCache),
    rng_buffers: zml.Bufferized(zml.Tensor.Rng),

    decode_runner: inference.KernelExe.Runner,

    tokenizer: zml.tokenizer.Tokenizer,
    generated_token_buf: zml.Buffer,
    generated_token_slice: zml.Slice,
    decode_last_real_pos_buf: zml.Buffer,
    seqlen: u32,
    eos_token_id: u32,
    special_tokens: model.Model.SpecialTokens,
    think_start: ?u32,
    think_end: ?u32,

    max_n_visual: u32,
    next_decode_pos: u32 = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        tokenizer: zml.tokenizer.Tokenizer,
        compiled_model: *const inference.CompiledModel,
        model_buffers: *model.Buffers,
    ) !Session {
        var kv_cache_buffers = try compiled_model.params.kv_cache.initBuffer(io, platform, compiled_model.params.shardings.model);
        errdefer model.KvCache.deinitBuffer(&kv_cache_buffers);

        const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
        var rng_buffers = try zml.Tensor.Rng.initBuffer(io, platform, .replicated, seed);
        errdefer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

        var decode_runner = try compiled_model.decode.initRunner(allocator, io, platform, model_buffers);
        errdefer decode_runner.deinit(allocator);

        var generated_token_buf = try zml.Buffer.uninitialized(io, platform, compiled_model.params.decode_token.shape(), .replicated, .{});
        errdefer generated_token_buf.deinit();

        const generated_token_slice = try zml.Slice.alloc(allocator, compiled_model.params.decode_token.shape());
        errdefer generated_token_slice.free(allocator);

        var decode_last_real_pos_buf = try zml.Buffer.scalar(io, platform, @as(u32, 0), .u32);
        errdefer decode_last_real_pos_buf.deinit();

        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .model_buffers = model_buffers,
            .compiled_model = compiled_model,
            .kv_cache_buffers = kv_cache_buffers,
            .rng_buffers = rng_buffers,
            .decode_runner = decode_runner,
            .tokenizer = tokenizer,
            .generated_token_buf = generated_token_buf,
            .generated_token_slice = generated_token_slice,
            .decode_last_real_pos_buf = decode_last_real_pos_buf,
            .seqlen = compiled_model.params.max_seqlen,
            .eos_token_id = compiled_model.loaded_model.inner.special_tokens.im_end_token_id,
            .special_tokens = compiled_model.loaded_model.inner.special_tokens,
            .think_start = tokenizer.tokenId("<think>"),
            .think_end = tokenizer.tokenId("</think>"),
            .max_n_visual = compiled_model.params.max_n_visual,
        };
    }

    pub fn deinit(self: *Session) void {
        model.KvCache.deinitBuffer(&self.kv_cache_buffers);
        zml.Tensor.Rng.deinitBuffer(&self.rng_buffers);
        self.decode_runner.deinit(self.allocator);
        self.generated_token_buf.deinit();
        self.generated_token_slice.free(self.allocator);
        self.decode_last_real_pos_buf.deinit();
    }

    pub fn tokenizePrompt(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return self.tokenizePromptMultimodal(allocator, prompt, null);
    }

    pub fn tokenizePromptMultimodal(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8, pp_image: ?PreprocessedImage) ![]const u32 {
        const config = &self.compiled_model.loaded_model.parsed_config.value;
        const merge: i64 = config.vision_config.spatial_merge_size;

        const n_visual_actual: usize = if (pp_image) |pi|
            @intCast(@divExact(pi.grid_h, merge) * @divExact(pi.grid_w, merge))
        else
            0;

        return tokenizeChatPromptVisual(allocator, self.tokenizer, prompt, self.special_tokens, n_visual_actual, true);
    }

    pub fn tokenizeTurn(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return tokenizeChatPromptVisual(allocator, self.tokenizer, prompt, self.special_tokens, 0, false);
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        return self.runPrefillMultimodal(all_tokens, null);
    }

    pub fn runPrefillMultimodal(self: *Session, all_tokens: []const u32, pp_image: ?PreprocessedImage) !void {
        const params = &self.compiled_model.params;
        const config = &self.compiled_model.loaded_model.parsed_config.value;
        const max_seqlen: usize = @intCast(params.max_seqlen);
        const max_n_visual: usize = @intCast(params.max_n_visual);
        const merge: i64 = config.vision_config.spatial_merge_size;

        const n_visual_actual: usize = if (pp_image) |pi|
            @intCast(@divExact(pi.grid_h, merge) * @divExact(pi.grid_w, merge))
        else
            0;
        const grid_h_merged: usize = if (pp_image) |pi| @intCast(@divExact(pi.grid_h, merge)) else 0;
        const grid_w_merged: usize = if (pp_image) |pi| @intCast(@divExact(pi.grid_w, merge)) else 0;

        // Scan all_tokens for the first image_pad token to locate the visual region.
        // For text-only, visual_start == all_tokens.len and n_suffix == 0.
        const visual_start: usize = if (pp_image != null)
            try findFirstToken(all_tokens, self.special_tokens.image_pad_token_id)
        else
            all_tokens.len;

        const n_prefix: usize = visual_start;
        const n_suffix: usize = all_tokens.len - n_prefix - n_visual_actual;
        const last_real_pos: u32 = @intCast(all_tokens.len - 1);

        const prefill_tokens_slice: zml.Slice = try .alloc(self.allocator, params.prefill_tokens.shape());
        defer prefill_tokens_slice.free(self.allocator);
        const prefill_tokens = prefill_tokens_slice.items(u32);
        @memset(prefill_tokens, self.special_tokens.end_of_text_token_id);
        @memcpy(prefill_tokens[0..all_tokens.len], all_tokens);

        const replicated_sharding: zml.Sharding = .replicated;

        var prefill_tokens_buf: zml.Buffer = try .fromSlice(self.io, self.platform, prefill_tokens_slice, replicated_sharding);
        defer prefill_tokens_buf.deinit();

        const position_ids_slice: zml.Slice = try .alloc(self.allocator, params.position_ids.shape());
        defer position_ids_slice.free(self.allocator);
        buildPositionIdsSlice(position_ids_slice.items(i64), max_seqlen, n_prefix, grid_h_merged, grid_w_merged, n_visual_actual, visual_start, n_suffix);
        var position_ids_buf: zml.Buffer = try .fromSlice(self.io, self.platform, position_ids_slice, replicated_sharding);
        defer position_ids_buf.deinit();

        const visual_scatter_idx_slice: zml.Slice = try .alloc(self.allocator, params.visual_scatter_idx.shape());
        defer visual_scatter_idx_slice.free(self.allocator);
        const visual_scatter_idx = visual_scatter_idx_slice.items(i32);
        for (0..n_visual_actual) |i| visual_scatter_idx[i] = @intCast(visual_start + i);
        for (n_visual_actual..max_n_visual) |i| visual_scatter_idx[i] = @intCast(max_seqlen - 1);
        var visual_scatter_idx_buf: zml.Buffer = try .fromSlice(self.io, self.platform, visual_scatter_idx_slice, replicated_sharding);
        defer visual_scatter_idx_buf.deinit();

        var token_index_buf: zml.Buffer = try .scalar(self.io, self.platform, @as(u32, 0), .u32);
        defer token_index_buf.deinit();
        var last_real_pos_buf: zml.Buffer = try .scalar(self.io, self.platform, last_real_pos, .u32);
        defer last_real_pos_buf.deinit();

        // Zero visual_embeds + deepstack buffers; vision exe overwrites these if image given.
        const visual_embeds_slice: zml.Slice = try .alloc(self.allocator, params.visual_embeds.shape());
        defer visual_embeds_slice.free(self.allocator);
        @memset(visual_embeds_slice.items(u8), 0);
        var visual_embeds_buf: zml.Buffer = try .fromSlice(self.io, self.platform, visual_embeds_slice, replicated_sharding);
        defer visual_embeds_buf.deinit();
        var deepstack_buf_0: zml.Buffer = try .fromSlice(self.io, self.platform, visual_embeds_slice, replicated_sharding);
        defer deepstack_buf_0.deinit();
        var deepstack_buf_1: zml.Buffer = try .fromSlice(self.io, self.platform, visual_embeds_slice, replicated_sharding);
        defer deepstack_buf_1.deinit();
        var deepstack_buf_2: zml.Buffer = try .fromSlice(self.io, self.platform, visual_embeds_slice, replicated_sharding);
        defer deepstack_buf_2.deinit();

        if (pp_image) |pi| {
            const n_actual_patches: usize = pi.h_pos.len;

            const patches_slice: zml.Slice = try .alloc(self.allocator, params.patches.shape());
            defer patches_slice.free(self.allocator);
            @memset(patches_slice.items(u8), 0);
            @memcpy(patches_slice.items(f32)[0..pi.patches.len], pi.patches);
            var patches_buf: zml.Buffer = try .fromSlice(self.io, self.platform, patches_slice, replicated_sharding);
            defer patches_buf.deinit();

            const h_pos_slice: zml.Slice = try .alloc(self.allocator, params.h_pos.shape());
            defer h_pos_slice.free(self.allocator);
            @memset(h_pos_slice.items(u8), 0);
            @memcpy(h_pos_slice.items(i64)[0..pi.h_pos.len], pi.h_pos);
            var h_pos_buf: zml.Buffer = try .fromSlice(self.io, self.platform, h_pos_slice, replicated_sharding);
            defer h_pos_buf.deinit();

            const w_pos_slice: zml.Slice = try .alloc(self.allocator, params.w_pos.shape());
            defer w_pos_slice.free(self.allocator);
            @memset(w_pos_slice.items(u8), 0);
            @memcpy(w_pos_slice.items(i64)[0..pi.w_pos.len], pi.w_pos);
            var w_pos_buf: zml.Buffer = try .fromSlice(self.io, self.platform, w_pos_slice, replicated_sharding);
            defer w_pos_buf.deinit();

            const attn_mask_v_slice: zml.Slice = try .alloc(self.allocator, params.attn_mask_v.shape());
            defer attn_mask_v_slice.free(self.allocator);
            const attn_mask_v = attn_mask_v_slice.items(f32);
            for (0..n_actual_patches) |i| attn_mask_v[i] = 0.0;
            for (n_actual_patches..@intCast(params.max_patches)) |i| attn_mask_v[i] = -1e9;
            var attn_mask_v_buf: zml.Buffer = try .fromSlice(self.io, self.platform, attn_mask_v_slice, replicated_sharding);
            defer attn_mask_v_buf.deinit();

            const visual_mask_slice: zml.Slice = try .alloc(self.allocator, params.visual_mask.shape());
            defer visual_mask_slice.free(self.allocator);
            const visual_mask = visual_mask_slice.items(f32);
            for (0..n_visual_actual) |i| visual_mask[i] = 1.0;
            for (n_visual_actual..max_n_visual) |i| visual_mask[i] = 0.0;
            var visual_mask_buf: zml.Buffer = try .fromSlice(self.io, self.platform, visual_mask_slice, replicated_sharding);
            defer visual_mask_buf.deinit();

            var grid_h_buf: zml.Buffer = try .scalar(self.io, self.platform, pi.grid_h, .i64);
            defer grid_h_buf.deinit();
            var grid_w_buf: zml.Buffer = try .scalar(self.io, self.platform, pi.grid_w, .i64);
            defer grid_w_buf.deinit();

            var vis_args = try self.compiled_model.vision_exe.args(self.allocator);
            defer vis_args.deinit(self.allocator);
            vis_args.bake(self.model_buffers.vision_model);
            var vis_results = try self.compiled_model.vision_exe.results(self.allocator);
            defer vis_results.deinit(self.allocator);

            log.info("running vision encoder...", .{});
            vis_args.set(.{ patches_buf, h_pos_buf, w_pos_buf, attn_mask_v_buf, visual_mask_buf, grid_h_buf, grid_w_buf });
            self.compiled_model.vision_exe.callOpts(self.io, vis_args, &vis_results, .{ .wait = true });

            const VisionReturn = zml.Bufferized(zml.stdx.meta.FnReturn(model.VisionModel.paddedVisionForward));
            const vis_out = vis_results.get(VisionReturn);
            visual_embeds_buf.deinit();
            visual_embeds_buf = vis_out[0];
            deepstack_buf_0.deinit();
            deepstack_buf_0 = vis_out[1];
            deepstack_buf_1.deinit();
            deepstack_buf_1 = vis_out[2];
            deepstack_buf_2.deinit();
            deepstack_buf_2 = vis_out[3];
        }

        log.info("running prefill...", .{});
        try self.compiled_model.prefill.run(.{
            .allocator = self.allocator,
            .io = self.io,
            .platform = self.platform,
            .model_buffers = self.model_buffers,
            .tokens_buf = &prefill_tokens_buf,
            .position_ids_buf = &position_ids_buf,
            .token_index_buf = &token_index_buf,
            .last_real_pos_buf = &last_real_pos_buf,
            .generated_token_buf = &self.generated_token_buf,
            .kv_cache_buffers = &self.kv_cache_buffers,
            .rng_buffers = &self.rng_buffers,
            .visual_embeds_buf = &visual_embeds_buf,
            .visual_scatter_idx_buf = &visual_scatter_idx_buf,
            .deepstack_bufs = .{ &deepstack_buf_0, &deepstack_buf_1, &deepstack_buf_2 },
        });

        self.next_decode_pos = @intCast(position_ids_slice.items(i64)[all_tokens.len - 1] + 1);

        const next_token = try self.generated_token_buf.getValue(u32, self.io);
        self.generated_token_slice.items(u32)[0] = next_token;
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
        var decoder = try self.tokenizer.decoder();
        defer decoder.deinit();

        const out_tokens_buffer: []u8 = try self.allocator.alloc(u8, 1024);
        defer self.allocator.free(out_tokens_buffer);
        const replicated_sharding: zml.Sharding = .replicated;

        var token_index_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, @intCast(all_tokens.items.len)), .u32);
        defer token_index_buffer.deinit();

        var decode_pos: u32 = self.next_decode_pos;

        generation: while (true) {
            const token_id = self.generated_token_slice.items(u32)[0];
            if (token_id == self.eos_token_id) break :generation;

            const token = try decoder.feedOne(token_id, out_tokens_buffer);
            if (self.think_start) |think_start| if (token_id == think_start) {
                try stdout.writeAll("\x1b[2m");
            };
            try stdout.writeAll(token);
            if (self.think_end) |think_end| if (token_id == think_end) {
                try stdout.writeAll("\x1b[0m");
            };
            try stdout.flush();

            try all_tokens.append(self.allocator, token_id);
            if (all_tokens.items.len >= self.seqlen) break :generation;

            const pos_arr = [_]i64{ decode_pos, decode_pos, decode_pos };
            var decode_pos_buf = try zml.Buffer.fromBytes(
                self.io,
                self.platform,
                self.compiled_model.params.decode_position_ids.shape(),
                replicated_sharding,
                std.mem.sliceAsBytes(&pos_arr),
            );
            defer decode_pos_buf.deinit();

            try self.decode_runner.run(.{
                .allocator = self.allocator,
                .io = self.io,
                .platform = self.platform,
                .model_buffers = self.model_buffers,
                .tokens_buf = &self.generated_token_buf,
                .position_ids_buf = &decode_pos_buf,
                .token_index_buf = &token_index_buffer,
                .last_real_pos_buf = &self.decode_last_real_pos_buf,
                .generated_token_buf = &self.generated_token_buf,
                .kv_cache_buffers = &self.kv_cache_buffers,
                .rng_buffers = &self.rng_buffers,
                .visual_embeds_buf = undefined,
                .visual_scatter_idx_buf = undefined,
                .deepstack_bufs = undefined,
            });

            try self.generated_token_buf.toSlice(self.io, self.generated_token_slice);

            decode_pos += 1;
            const new_idx = try zml.Buffer.scalar(self.io, self.platform, @as(u32, @intCast(all_tokens.items.len)), .u32);
            token_index_buffer.deinit();
            token_index_buffer = new_idx;
        }

        try stdout.writeAll(try decoder.finalize(out_tokens_buffer));
        try stdout.flush();
    }

    pub fn preprocessImage(self: *const Session, allocator: std.mem.Allocator, bytes: []const u8) !PreprocessedImage {
        const vcfg = self.compiled_model.loaded_model.parsed_config.value.vision_config;
        return try image.preprocess(
            allocator,
            bytes,
            self.compiled_model.params.max_patches,
            vcfg.patch_size,
            vcfg.temporal_patch_size,
            vcfg.in_channels,
            vcfg.spatial_merge_size,
        );
    }
};

fn buildPositionIdsSlice(
    out: []i64,
    max_seqlen: usize,
    n_prefix: usize,
    grid_h_merged: usize,
    grid_w_merged: usize,
    n_visual_actual: usize,
    visual_start: usize,
    n_suffix: usize,
) void {
    for (0..n_prefix) |i| {
        const p: i64 = @intCast(i);
        out[0 * max_seqlen + i] = p;
        out[1 * max_seqlen + i] = p;
        out[2 * max_seqlen + i] = p;
    }
    const np: i64 = @intCast(n_prefix);
    for (0..n_visual_actual) |vi| {
        const row: i64 = @intCast(vi / grid_w_merged);
        const col: i64 = @intCast(vi % grid_w_merged);
        const sp = visual_start + vi;
        out[0 * max_seqlen + sp] = np;
        out[1 * max_seqlen + sp] = np + row;
        out[2 * max_seqlen + sp] = np + col;
    }
    const n_start: i64 = np + @as(i64, @intCast(@max(grid_h_merged, grid_w_merged)));
    var suf_pos: i64 = n_start;
    const real_start = visual_start + n_visual_actual;
    const real_end = real_start + n_suffix;
    for (real_start..max_seqlen) |i| {
        out[0 * max_seqlen + i] = suf_pos;
        out[1 * max_seqlen + i] = suf_pos;
        out[2 * max_seqlen + i] = suf_pos;
        if (i < real_end) suf_pos += 1;
    }
}

fn findFirstToken(tokens: []const u32, id: u32) error{MissingToken}!usize {
    for (tokens, 0..) |t, i| if (t == id) return i;
    return error.MissingToken;
}

fn tokenizeChatPromptVisual(
    allocator: std.mem.Allocator,
    tokenizer: zml.tokenizer.Tokenizer,
    prompt: []const u8,
    special_tokens: model.Model.SpecialTokens,
    n_visual_actual: usize,
    is_first_turn: bool,
) ![]const u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const im_start = tokenizer.tokenId("<|im_start|>") orelse special_tokens.im_start_token_id;
    const im_end = tokenizer.tokenId("<|im_end|>") orelse special_tokens.im_end_token_id;
    const vstart = tokenizer.tokenId("<|vision_start|>") orelse special_tokens.vision_start_token_id;
    const vend = tokenizer.tokenId("<|vision_end|>") orelse special_tokens.vision_end_token_id;
    const ipad = tokenizer.tokenId("<|image_pad|>") orelse special_tokens.image_pad_token_id;
    const newline = tokenizer.tokenId("\\n") orelse return error.NoSuchToken;

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, 128);
    if (!is_first_turn) {
        try tokens.appendSlice(allocator, &.{ im_end, newline });
    }

    try tokens.append(allocator, im_start);
    const user_tokens = try encoder.encodeAlloc(allocator, "user\n");
    defer allocator.free(user_tokens);
    try tokens.appendSlice(allocator, user_tokens);

    if (n_visual_actual > 0) {
        try tokens.append(allocator, vstart);
        try tokens.appendNTimes(allocator, ipad, n_visual_actual);
        try tokens.append(allocator, vend);
    }

    const prompt_tokens = try encoder.encodeAlloc(allocator, prompt);
    defer allocator.free(prompt_tokens);
    try tokens.appendSlice(allocator, prompt_tokens);

    try tokens.appendSlice(allocator, &.{ im_end, newline, im_start });
    const assistant_tokens = try encoder.encodeAlloc(allocator, "assistant\n");
    defer allocator.free(assistant_tokens);
    try tokens.appendSlice(allocator, assistant_tokens);

    return tokens.toOwnedSlice(allocator);
}
