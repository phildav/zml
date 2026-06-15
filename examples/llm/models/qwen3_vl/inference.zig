const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const model = @import("model.zig");

const log = std.log.scoped(.qwen3_vl);
const Phase = common.Phase;

pub const CompilationParameters = struct {
    patches: zml.Tensor,
    h_pos: zml.Tensor,
    w_pos: zml.Tensor,
    attn_mask_v: zml.Tensor,
    visual_mask: zml.Tensor,
    grid_h: zml.Tensor,
    grid_w: zml.Tensor,

    prefill_tokens: zml.Tensor,
    position_ids: zml.Tensor,
    visual_embeds: zml.Tensor,
    deepstack: [3]zml.Tensor,
    visual_scatter_idx: zml.Tensor,
    last_real_pos: zml.Tensor,

    decode_token: zml.Tensor,
    decode_position_ids: zml.Tensor,

    token_index: zml.Tensor,
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
            .deepstack = .{
                .init(.{ .n = max_n_visual, .d = hidden }, dtype),
                .init(.{ .n = max_n_visual, .d = hidden }, dtype),
                .init(.{ .n = max_n_visual, .d = hidden }, dtype),
            },
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

pub const CompilationOptions = CompilationParameters;

pub const Args = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_buffers: *model.Buffers,
    tokens_buf: *zml.Buffer,
    position_ids_buf: *zml.Buffer,
    token_index_buf: *zml.Buffer,
    last_real_pos_buf: *zml.Buffer,
    generated_token_buf: *zml.Buffer,
    kv_cache_buffers: *zml.Bufferized(model.KvCache),
    rng_buffers: *zml.Bufferized(zml.Tensor.Rng),
    visual_embeds_buf: *zml.Buffer,
    visual_scatter_idx_buf: *zml.Buffer,
    deepstack_bufs: [3]*zml.Buffer,
};

pub const CompiledModel = struct {
    loaded_model: *const model.LoadedModel,
    vision_exe: zml.Exe,
    prefill: KernelExe,
    decode: KernelExe,
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
        const vision_exe = try compileVision(allocator, io, platform, qwen_model, parameters, progress);
        errdefer vision_exe.deinit();

        const prefill = try KernelExe.init(
            allocator,
            io,
            platform,
            qwen_model,
            parameters,
            @intCast(parameters.prefill_tokens.dim(.s)),
            .prefill,
            progress,
        );
        errdefer prefill.deinit();

        const decode = try KernelExe.init(
            allocator,
            io,
            platform,
            qwen_model,
            parameters,
            @intCast(parameters.decode_token.dim(.s)),
            .decode,
            progress,
        );
        errdefer decode.deinit();

        return .{
            .loaded_model = loaded_model,
            .vision_exe = vision_exe,
            .prefill = prefill,
            .decode = decode,
            .params = parameters,
        };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.vision_exe.deinit();
        self.prefill.deinit();
        self.decode.deinit();
    }
};

pub const Inference = CompiledModel;

pub const KernelExe = struct {
    composed: ComposedKernelExe,

    pub const Runner = struct {
        exe: *const ComposedKernelExe,
        embed_args: zml.exe.Exe.Arguments,
        embed_results: zml.exe.Exe.Results,
        layers: Layers,
        sampler_args: zml.exe.Exe.Arguments,
        sampler_results: zml.exe.Exe.Results,

        const Layers = struct {
            args: []zml.exe.Exe.Arguments,
            results: []zml.exe.Exe.Results,
            kv_cache_indices: []zml.Buffer,

            fn init(
                allocator: std.mem.Allocator,
                io: std.Io,
                platform: *const zml.Platform,
                exe: *const ComposedKernelExe,
                model_buffers: *model.Buffers,
            ) !Layers {
                const args = try allocator.alloc(zml.exe.Exe.Arguments, model_buffers.text_model.layers.len);
                errdefer allocator.free(args);

                const results = try allocator.alloc(zml.exe.Exe.Results, model_buffers.text_model.layers.len);
                errdefer allocator.free(results);

                const kv_cache_indices = try allocator.alloc(zml.Buffer, model_buffers.text_model.layers.len);
                errdefer allocator.free(kv_cache_indices);

                var initialized_args: usize = 0;
                errdefer {
                    for (args[0..initialized_args]) |*exe_args| {
                        exe_args.deinit(allocator);
                    }
                }

                var initialized_results: usize = 0;
                errdefer {
                    for (results[0..initialized_results]) |*exe_results| {
                        exe_results.deinit(allocator);
                    }
                }

                var initialized_kv_cache_indices: usize = 0;
                errdefer {
                    for (kv_cache_indices[0..initialized_kv_cache_indices]) |*kv_cache_index| {
                        kv_cache_index.deinit();
                    }
                }

                for (model_buffers.text_model.layers, 0..) |layer_bufs, i| {
                    args[i] = try exe.layer.args(allocator);
                    initialized_args = i + 1;
                    args[i].bake(layer_bufs);

                    results[i] = try exe.layer.results(allocator);
                    initialized_results = i + 1;

                    kv_cache_indices[i] = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(i)), .u32);
                    initialized_kv_cache_indices = i + 1;
                }

                return .{ .args = args, .results = results, .kv_cache_indices = kv_cache_indices };
            }

            fn deinit(self: *Layers, allocator: std.mem.Allocator) void {
                for (self.args) |*exe_args| {
                    exe_args.deinit(allocator);
                }
                allocator.free(self.args);

                for (self.results) |*exe_results| {
                    exe_results.deinit(allocator);
                }
                allocator.free(self.results);

                for (self.kv_cache_indices) |*kv_cache_index| {
                    kv_cache_index.deinit();
                }
                allocator.free(self.kv_cache_indices);
            }
        };

        pub fn init(
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            exe: *const ComposedKernelExe,
            model_buffers: *model.Buffers,
        ) !Runner {
            var embed_args = try exe.embed_tokens.args(allocator);
            errdefer embed_args.deinit(allocator);
            embed_args.bake(ComposedKernelExe.embedTokensBuffers(model_buffers));

            var embed_results = try exe.embed_tokens.results(allocator);
            errdefer embed_results.deinit(allocator);

            var layers = try Layers.init(allocator, io, platform, exe, model_buffers);
            errdefer layers.deinit(allocator);

            var sampler_args = try exe.sampler.args(allocator);
            errdefer sampler_args.deinit(allocator);
            sampler_args.bake(ComposedKernelExe.samplerBuffers(model_buffers));

            var sampler_results = try exe.sampler.results(allocator);
            errdefer sampler_results.deinit(allocator);

            return .{
                .exe = exe,
                .embed_args = embed_args,
                .embed_results = embed_results,
                .layers = layers,
                .sampler_args = sampler_args,
                .sampler_results = sampler_results,
            };
        }

        pub fn deinit(self: *Runner, allocator: std.mem.Allocator) void {
            self.embed_args.deinit(allocator);
            self.embed_results.deinit(allocator);
            self.layers.deinit(allocator);
            self.sampler_args.deinit(allocator);
            self.sampler_results.deinit(allocator);
        }

        pub fn run(self: *Runner, args: Args) !void {
            var hidden_buf: zml.Buffer = b: {
                self.embed_args.set(.{args.tokens_buf});
                self.exe.embed_tokens.call(self.embed_args, &self.embed_results);
                break :b self.embed_results.get(zml.Buffer);
            };
            defer hidden_buf.deinit();

            for (
                self.layers.args,
                self.layers.results,
                self.layers.kv_cache_indices,
            ) |*exe_args, *results, *kv_cache_index_buf| {
                self.exe.runLayer(exe_args, results, args, &hidden_buf, kv_cache_index_buf);
            }

            self.exe.runSampler(&self.sampler_args, &self.sampler_results, args, &hidden_buf);
        }
    };

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        qwen_model: model.Model,
        parameters: CompilationOptions,
        seqlen: usize,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !KernelExe {
        return .{
            .composed = try .init(allocator, io, platform, qwen_model, parameters, seqlen, phase, progress),
        };
    }

    pub fn deinit(self: KernelExe) void {
        self.composed.deinit();
    }

    pub fn run(self: *const KernelExe, args: Args) !void {
        try self.composed.run(args);
    }

    pub fn initRunner(
        self: *const KernelExe,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        model_buffers: *model.Buffers,
    ) !Runner {
        return .init(allocator, io, platform, &self.composed, model_buffers);
    }
};

const ComposedKernelExe = struct {
    embed_tokens: zml.Exe,
    layer: zml.Exe,
    sampler: zml.Exe,
    deepstack_inject: ?zml.Exe,

    const EmbedTokens = struct {
        embed_tokens: zml.nn.TokenEmbedding,

        pub fn forward(self: EmbedTokens, tokens: zml.Tensor) zml.Tensor {
            return self.embed_tokens.weight.gather(.{ .voc = tokens.withPartialTags(.{.s}) }, .{});
        }
    };

    const EmbedAndScatter = struct {
        embed_tokens: zml.nn.TokenEmbedding,

        pub fn forward(
            self: EmbedAndScatter,
            tokens: zml.Tensor,
            visual_embeds: zml.Tensor,
            scatter_idx: zml.Tensor,
        ) zml.Tensor {
            const x = self.embed_tokens.weight.gather(.{ .voc = tokens.withPartialTags(.{.s}) }, .{});
            return x.scatterSlices(.{ .s = scatter_idx }, visual_embeds, .{ .update_fn = zml.Tensor.ScatterOpts.override });
        }
    };

    const DeepstackInject = struct {
        pub fn forward(_: @This(), hidden: zml.Tensor, deepstack: zml.Tensor, scatter_idx: zml.Tensor) zml.Tensor {
            return hidden.scatterSlices(.{ .s = scatter_idx }, deepstack, .{ .update_fn = zml.Tensor.ScatterOpts.increment });
        }
    };

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        qwen_model: model.Model,
        parameters: CompilationOptions,
        seqlen: usize,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !ComposedKernelExe {
        const embed_tokens = try ComposedKernelExe.compileEmbedTokens(allocator, io, platform, qwen_model, parameters, seqlen, phase, progress);
        errdefer embed_tokens.deinit();

        const layer = try ComposedKernelExe.compileLayer(allocator, io, platform, qwen_model, parameters, seqlen, phase, progress);
        errdefer layer.deinit();

        const sampler = try ComposedKernelExe.compileSampler(allocator, io, platform, qwen_model, parameters, seqlen, phase, progress);
        errdefer sampler.deinit();

        const deepstack_inject: ?zml.Exe = switch (phase) {
            .prefill => b: {
                const exe = try ComposedKernelExe.compileDeepstackInject(allocator, io, platform, qwen_model, parameters, progress);
                break :b exe;
            },
            .decode => null,
        };
        errdefer if (deepstack_inject) |exe| exe.deinit();

        return .{
            .embed_tokens = embed_tokens,
            .layer = layer,
            .sampler = sampler,
            .deepstack_inject = deepstack_inject,
        };
    }

    fn deinit(self: ComposedKernelExe) void {
        self.embed_tokens.deinit();
        self.layer.deinit();
        self.sampler.deinit();
        if (self.deepstack_inject) |exe| exe.deinit();
    }

    fn run(self: *const ComposedKernelExe, args: Args) !void {
        var hidden_buf: zml.Buffer = b: {
            var exe_args = try self.embed_tokens.args(args.allocator);
            defer exe_args.deinit(args.allocator);

            var results = try self.embed_tokens.results(args.allocator);
            defer results.deinit(args.allocator);

            exe_args.bake(ComposedKernelExe.embedAndScatterBuffers(args.model_buffers));
            exe_args.set(.{ args.tokens_buf, args.visual_embeds_buf, args.visual_scatter_idx_buf });

            self.embed_tokens.call(exe_args, &results);

            break :b results.get(zml.Buffer);
        };
        defer hidden_buf.deinit();

        for (args.model_buffers.text_model.layers, 0..) |layer_bufs, i| {
            var exe_args = try self.layer.args(args.allocator);
            defer exe_args.deinit(args.allocator);

            var results = try self.layer.results(args.allocator);
            defer results.deinit(args.allocator);

            var kv_cache_index_buf: zml.Buffer = try .scalar(args.io, args.platform, @as(u32, @intCast(i)), .u32);
            defer kv_cache_index_buf.deinit();

            exe_args.bake(layer_bufs);

            self.runLayer(&exe_args, &results, args, &hidden_buf, &kv_cache_index_buf);

            if (i < 3) {
                if (self.deepstack_inject) |ds_exe| {
                    var ds_args = try ds_exe.args(args.allocator);
                    defer ds_args.deinit(args.allocator);

                    var ds_results = try ds_exe.results(args.allocator);
                    defer ds_results.deinit(args.allocator);

                    self.runDeepstackInject(&ds_args, &ds_results, &hidden_buf, args.deepstack_bufs[i], args.visual_scatter_idx_buf);
                }
            }
        }

        {
            var exe_args = try self.sampler.args(args.allocator);
            defer exe_args.deinit(args.allocator);

            var results = try self.sampler.results(args.allocator);
            defer results.deinit(args.allocator);

            exe_args.bake(ComposedKernelExe.samplerBuffers(args.model_buffers));

            self.runSampler(&exe_args, &results, args, &hidden_buf);
        }
    }

    fn runLayer(
        self: *const ComposedKernelExe,
        exe_args: *zml.exe.Exe.Arguments,
        results: *zml.exe.Exe.Results,
        args: Args,
        hidden_buf: *zml.Buffer,
        layer_index_buf: *zml.Buffer,
    ) void {
        const layer_kv: zml.Bufferized(model.KvCache) = .{
            .k = args.kv_cache_buffers.k,
            .v = args.kv_cache_buffers.v,
            .layer_index = layer_index_buf.*,
        };
        exe_args.set(.{ hidden_buf, args.position_ids_buf, args.token_index_buf, layer_kv });
        self.layer.call(exe_args.*, results);

        var new_hidden, var new_kv = results.get(struct { zml.Buffer, zml.Bufferized(model.KvCache) });
        ComposedKernelExe.replaceBuffer(hidden_buf, &new_hidden);
        ComposedKernelExe.replaceBuffer(&args.kv_cache_buffers.k, &new_kv.k);
        ComposedKernelExe.replaceBuffer(&args.kv_cache_buffers.v, &new_kv.v);
        ComposedKernelExe.releaseBuffer(layer_index_buf.*, &new_kv.layer_index);
    }

    fn runSampler(
        self: *const ComposedKernelExe,
        exe_args: *zml.exe.Exe.Arguments,
        results: *zml.exe.Exe.Results,
        args: Args,
        hidden_buf: *zml.Buffer,
    ) void {
        exe_args.set(.{ hidden_buf, args.rng_buffers, args.last_real_pos_buf });
        self.sampler.call(exe_args.*, results);

        var new_token, var new_rng = results.get(struct { zml.Buffer, zml.Bufferized(zml.Tensor.Rng) });
        ComposedKernelExe.replaceBuffer(args.generated_token_buf, &new_token);
        ComposedKernelExe.replaceBuffer(&args.rng_buffers._state, &new_rng._state);
    }

    fn runDeepstackInject(
        self: *const ComposedKernelExe,
        exe_args: *zml.exe.Exe.Arguments,
        results: *zml.exe.Exe.Results,
        hidden_buf: *zml.Buffer,
        deepstack_buf: *zml.Buffer,
        scatter_idx_buf: *zml.Buffer,
    ) void {
        exe_args.set(.{ hidden_buf, deepstack_buf, scatter_idx_buf });
        (self.deepstack_inject orelse unreachable).call(exe_args.*, results);

        var new_hidden = results.get(zml.Buffer);
        ComposedKernelExe.replaceBuffer(hidden_buf, &new_hidden);
    }

    fn embedTokensBuffers(model_buffers: *const model.Buffers) zml.Bufferized(EmbedTokens) {
        return .{ .embed_tokens = model_buffers.text_model.embed_tokens };
    }

    fn embedAndScatterBuffers(model_buffers: *const model.Buffers) zml.Bufferized(EmbedAndScatter) {
        return .{ .embed_tokens = model_buffers.text_model.embed_tokens };
    }

    fn samplerBuffers(model_buffers: *const model.Buffers) zml.Bufferized(model.Sampler) {
        return .{ .norm = model_buffers.text_model.norm, .lm_head = model_buffers.lm_head };
    }

    fn replaceBuffer(dst: *zml.Buffer, src: *zml.Buffer) void {
        if (!ComposedKernelExe.sameBufferHandle(dst.*, src.*)) {
            dst.deinit();
        }
        dst.* = src.*;
    }

    fn releaseBuffer(expected: zml.Buffer, actual: *zml.Buffer) void {
        if (!ComposedKernelExe.sameBufferHandle(expected, actual.*)) {
            actual.deinit();
        }
    }

    fn sameBufferHandle(a: zml.Buffer, b: zml.Buffer) bool {
        if (a._shards.len != b._shards.len) return false;
        for (a._shards.constSlice(), b._shards.constSlice()) |a_shard, b_shard| {
            if (a_shard != b_shard) return false;
        }
        return true;
    }

    fn compileEmbedTokens(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        qwen_model: model.Model,
        parameters: CompilationOptions,
        seqlen: usize,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !zml.Exe {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage("embed tokens"), 1);
        defer node.end();

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "embed tokens", io, from);

        switch (phase) {
            .prefill => {
                const tokens: zml.Tensor = .init(.{ .s = seqlen }, .u32);
                const visual_embeds = parameters.visual_embeds;
                const scatter_idx = parameters.visual_scatter_idx;
                return platform.compile(
                    allocator,
                    io,
                    EmbedAndScatter{ .embed_tokens = qwen_model.text_model.embed_tokens },
                    .forward,
                    .{ tokens, visual_embeds, scatter_idx },
                    .{
                        .shardings = &parameters.shardings.all(),
                        .program_name = phase.programName("qwen3_vl", "embed_tokens"),
                    },
                );
            },
            .decode => {
                const tokens: zml.Tensor = .init(.{ .s = 1 }, .u32);
                return platform.compile(
                    allocator,
                    io,
                    EmbedTokens{ .embed_tokens = qwen_model.text_model.embed_tokens },
                    .forward,
                    .{tokens},
                    .{
                        .shardings = &parameters.shardings.all(),
                        .program_name = phase.programName("qwen3_vl", "embed_tokens"),
                    },
                );
            },
        }
    }

    fn compileLayer(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        qwen_model: model.Model,
        parameters: CompilationOptions,
        seqlen: usize,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !zml.Exe {
        const Layer = struct {
            layer: model.TransformerLayer,

            pub fn forward(
                self: @This(),
                hidden: zml.Tensor,
                position_ids: zml.Tensor,
                token_index: zml.Tensor,
                kv_cache: model.KvCache,
            ) struct { zml.Tensor, model.KvCache } {
                return self.layer.forward(hidden, position_ids, token_index, kv_cache);
            }
        };

        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage("transformer layer"), 1);
        defer node.end();

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "transformer layer", io, from);

        const dtype = qwen_model.text_model.embed_tokens.weight.dtype();
        const hidden_size = qwen_model.config.text_config.hidden_size;
        const hidden: zml.Tensor = .fromShape(zml.Shape.init(
            .{ .s = seqlen, .d = hidden_size },
            dtype,
        ));

        const position_ids = switch (phase) {
            .prefill => parameters.position_ids,
            .decode => parameters.decode_position_ids,
        };

        const kv_template = model.KvCache{
            .k = parameters.kv_cache.k,
            .v = parameters.kv_cache.v,
            .layer_index = .init(.{}, .u32),
        };

        return platform.compile(
            allocator,
            io,
            Layer{ .layer = qwen_model.text_model.layers[0] },
            .forward,
            .{ hidden, position_ids, parameters.token_index, kv_template },
            .{
                .shardings = &parameters.shardings.all(),
                .program_name = phase.programName("qwen3_vl", "layer"),
            },
        );
    }

    fn compileSampler(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        qwen_model: model.Model,
        parameters: CompilationOptions,
        seqlen: usize,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !zml.Exe {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage("sampler"), 1);
        defer node.end();

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "sampler", io, from);

        const dtype = qwen_model.text_model.embed_tokens.weight.dtype();
        const hidden_size = qwen_model.config.text_config.hidden_size;
        const hidden: zml.Tensor = .fromShape(zml.Shape.init(
            .{ .s = seqlen, .d = hidden_size },
            dtype,
        ));

        return platform.compile(
            allocator,
            io,
            qwen_model.sampler(),
            .sampleTokens,
            .{ hidden, parameters.rng, parameters.last_real_pos },
            .{
                .shardings = &parameters.shardings.all(),
                .program_name = phase.programName("qwen3_vl", "sampler"),
            },
        );
    }

    fn compileDeepstackInject(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        qwen_model: model.Model,
        parameters: CompilationOptions,
        progress: *std.Progress.Node,
    ) !zml.Exe {
        const phase: Phase = .prefill;

        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage("deepstack inject"), 1);
        defer node.end();

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "deepstack inject", io, from);

        const dtype = qwen_model.text_model.embed_tokens.weight.dtype();
        const hidden_size = qwen_model.config.text_config.hidden_size;
        const max_seqlen = parameters.max_seqlen;

        const hidden: zml.Tensor = .fromShape(zml.Shape.init(
            .{ .s = max_seqlen, .d = hidden_size },
            dtype,
        ));

        return platform.compile(
            allocator,
            io,
            DeepstackInject{},
            .forward,
            .{ hidden, parameters.deepstack[0], parameters.visual_scatter_idx },
            .{
                .shardings = &parameters.shardings.all(),
                .program_name = phase.programName("qwen3_vl", "deepstack_inject"),
            },
        );
    }
};

fn compileVision(
    allocator: std.mem.Allocator,
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
        allocator,
        io,
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
