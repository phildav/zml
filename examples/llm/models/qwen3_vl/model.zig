const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const inference = @import("inference.zig");

const log = std.log.scoped(.qwen3_vl);

pub const Config = struct {
    vision_config: VisionConfig,
    text_config: TextConfig,
};

pub const TextConfig = struct {
    // General
    num_hidden_layers: i64,
    hidden_size: i64,
    max_position_embeddings: i64,
    rms_norm_eps: f32,
    // Self attention
    head_dim: i64,
    num_attention_heads: i64,
    num_key_value_heads: i64,
    rope_theta: f32,
    rope_scaling: RopeScaling,
};

pub const VisionConfig = struct {
    depth: i64,
    hidden_size: i64,
    num_heads: i64,
    in_channels: i64,
    patch_size: i64,
    spatial_merge_size: i64,
    temporal_patch_size: i64,
    out_hidden_size: i64,
    num_position_embeddings: i64,
    deepstack_visual_indexes: [3]i64,
    initializer_range: f32,
    rope_theta: f32 = 10000.0, // hardcoded as in HF
    norm_eps:f32 = 1e-6, // hardcoded as in HF

    pub fn patchEmbedDim(self: VisionConfig) i64 {
        return self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size;
    }
};

pub const RopeScaling = struct {
    mrope_section: [3]i64,
};

pub const LoadedModel = struct {
    inner: Model,
    parsed_config: std.json.Parsed(Config),

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        store: zml.io.TensorStore.View,
        generation: common.GenerationOptions,
    ) !LoadedModel {
        const parsed_config = try common.parseConfig(Config, allocator, io, repo);
        errdefer parsed_config.deinit();

        const options: Model.GenOptions = .{
            .sampling_strategy = generation.sampling_strategy,
            .max_patches = generation.max_patches,
        };

        return .{
            .inner = try .init(allocator, store, parsed_config.value, options),
            .parsed_config = parsed_config,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
        self.parsed_config.deinit();
    }

    pub fn loadBuffers(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
        shardings: common.Shardings,
    ) !Buffers {
        progress.increaseEstimatedTotalItems(store.view().count());
        const now: std.Io.Timestamp = .now(io, .awake);
        var total_bytes: usize = 0;
        defer {
            const took = now.untilNow(io, .awake);
            const bytes_per_sec: u64 = @intFromFloat(
                @as(f64, @floatFromInt(total_bytes)) /
                    (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s),
            );
            log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });
        }

        const all_shardings = shardings.all();
        return zml.io.load(Model, &self.inner, allocator, io, platform, store, .{
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = progress,
            .shardings = &all_shardings,
            .parallelism = 16,
            .total_bytes = &total_bytes,
        });
    }

    pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        _ = self;
        Model.unloadBuffers(buffers, allocator);
    }

    pub fn compile(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        backend: zml.attention.attention.Backend,
        shardings: common.Shardings,
        seqlen: usize,
        progress: *std.Progress.Node,
    ) !inference.CompiledModel {
        _ = backend;
        const params = inference.CompilationParameters.init(
            self.inner, self.parsed_config.value,
            @intCast(seqlen),
            @intCast(self.inner.gen_options.max_patches),
            shardings
        );
        return inference.CompiledModel.init(allocator, io, platform, self, self.inner, params, progress);
    }
};

pub const Buffers = zml.Bufferized(Model);

pub const Model = struct {
    pub const GenOptions = struct { sampling_strategy: zml.nn.SamplingStrategy = .{}, max_patches: u32 = 1024};

    pub const SpecialTokens = struct {
        im_start_token_id: u32,
        im_end_token_id: u32,
        end_of_text_token_id: u32,
        vision_start_token_id: u32,
        vision_end_token_id: u32,
        image_pad_token_id: u32,
    };
    
    vision_model: VisionModel,
    text_model: TextModel,
    lm_head: zml.nn.Linear,

    config: Config,
    gen_options: GenOptions,
    special_tokens: SpecialTokens = .{
        .im_start_token_id = 151644,
        .im_end_token_id = 151645,
        .end_of_text_token_id = 151643,
        .vision_start_token_id = 151652,
        .vision_end_token_id = 151653,
        .image_pad_token_id = 151655,
    },

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, gen_options: GenOptions) !Model {
        const lm_head_prefix = b: {
            if (store.hasKey("lm_head.weight")) break :b "lm_head";
            break :b "model.language_model.embed_tokens";
        };
        const lm_head_weight = store.withPrefix(lm_head_prefix).createTensor(
            "weight", .{ .voc, .d }, .{ .voc = .model, .d = .replicated });

        return .{
            .vision_model = try VisionModel.init(
                allocator,
                store.withPrefix("model.visual"),
                config.vision_config
            ),
            .text_model = try TextModel.init(
                allocator,
                store.withPrefix("model.language_model"),
                config.text_config,
            ),
            .lm_head = .init(lm_head_weight, null, .d),
            .config = config,
            .gen_options = gen_options,
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        self.vision_model.deinit(allocator);
        self.text_model.deinit(allocator);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Model), allocator: std.mem.Allocator) void {
        VisionModel.unloadBuffers(&self.vision_model, allocator);
        TextModel.unloadBuffers(&self.text_model, allocator);
        self.lm_head.weight.deinit();
        if (self.lm_head.bias) |*b| b.deinit();
    }

    pub fn sampler(self: Model) Sampler {
        return .{
            .norm = self.text_model.norm,
            .lm_head = self.lm_head,
            .gen_options = self.gen_options,
        };
    }

};

pub const Sampler = struct {
    norm: RmsNorm,
    lm_head: zml.nn.Linear,
    gen_options: Model.GenOptions,

    pub fn sampleTokens(
        self: Sampler,
        out: zml.Tensor,        // [.s, .d]
        rng: zml.Tensor.Rng,
        last_real_pos: zml.Tensor, // [] u32 — sequence position to sample at
    ) struct { zml.Tensor, zml.Tensor.Rng } {
        const sliced = out.dynamicSlice(.{ .s = zml.Tensor.DynSlice{ .start = last_real_pos, .len = 1 } });
        const normed = self.norm.forward(sliced);        // [.s=1, .d]
        const logits = self.lm_head.forward(normed);    // [.s=1, .voc]
        const next_token, const new_rng = zml.nn.sampleTokens(logits, self.gen_options.sampling_strategy, rng);
        return .{ next_token.convert(.u32), new_rng };
    }
};


pub const TextModel = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    norm: RmsNorm,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: TextConfig) !TextModel {
        const layers = try allocator.alloc(TransformerLayer, @intCast(config.num_hidden_layers));
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            layer.* = .init(store.withPrefix("layers").withLayer(i), config);
        }

        return .{
            .embed_tokens = .{ .weight = store.createTensor("embed_tokens.weight", .{ .voc, .d }, .{ .voc = .replicated, .d = .model }) },
            .layers = layers,
            .norm = .init(store.withPrefix("norm"), config.rms_norm_eps),
        };
    }

    pub fn deinit(self: TextModel, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TextModel), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| {
            TransformerLayer.unloadBuffers(layer);
        }
        allocator.free(self.layers);
        RmsNorm.unloadBuffers(&self.norm);
    }

};

pub const TransformerLayer = struct {
    input_layernorm: RmsNorm,
    attn: SelfAttn,
    mlp: Mlp,
    post_attention_layernorm: RmsNorm,

    pub fn init(store: zml.io.TensorStore.View, config: TextConfig) TransformerLayer {
        return .{
            .input_layernorm = .init(store.withPrefix("input_layernorm"), config.rms_norm_eps),
            .attn = .init(store.withPrefix("self_attn"), config),
            .mlp = .init(store.withPrefix("mlp")),
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm"), config.rms_norm_eps),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        SelfAttn.unloadBuffers(&self.attn);
        Mlp.unloadBuffers(&self.mlp);
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
    }

    pub fn forward(
        self: TransformerLayer,
        x: zml.Tensor,
        position_ids: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache
    ) struct { zml.Tensor, KvCache } {
        const norm = self.input_layernorm.forward(x);
        // TODO reuseBuffer ?
        const attention_output, const updated_kv_cache = self.attn.forward(norm, position_ids, token_index, kv_cache);
        const x1 = attention_output.rename(.{ .q = .s, .dout = .d }).add(x);

        const normalized_hidden = self.post_attention_layernorm.forward(x1);
        const mlp_output = self.mlp.forward(normalized_hidden);

        return .{ mlp_output.add(x1), updated_kv_cache };
    }
};

pub const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    rotary_embed: TextRotaryEmbedding,

    pub fn init(store: zml.io.TensorStore.View, config: TextConfig) SelfAttn {
        return .{
            .q_proj = .init(store.createTensor("q_proj.weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }), null, .d),
            .k_proj = .init(store.createTensor("k_proj.weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }), null, .d),
            .v_proj = .init(store.createTensor("v_proj.weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }), null, .d),
            .o_proj = .init(store.createTensor("o_proj.weight", .{ .dout, .d }, .{ .dout = .replicated, .d = .model }), null, .d),
            .q_norm = .init(store.withPrefix("q_norm"), config.rms_norm_eps),
            .k_norm = .init(store.withPrefix("k_norm"), config.rms_norm_eps),
            .num_heads = config.num_attention_heads,
            .num_kv_heads = config.num_key_value_heads,
            .head_dim = config.head_dim,
            .rotary_embed = .init(config.head_dim, config.rope_theta, config.rope_scaling.mrope_section),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SelfAttn)) void {
        self.q_proj.weight.deinit();
        if (self.q_proj.bias) |*bias| bias.deinit();
        self.k_proj.weight.deinit();
        if (self.k_proj.bias) |*bias| bias.deinit();
        self.v_proj.weight.deinit();
        if (self.v_proj.bias) |*bias| bias.deinit();
        self.o_proj.weight.deinit();
        if (self.o_proj.bias) |*bias| bias.deinit();
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
    }

    pub fn forward(self: SelfAttn, x: zml.Tensor, position_ids: zml.Tensor, token_index: zml.Tensor, kv_cache: KvCache) struct { zml.Tensor, KvCache } {
        const orig_dtype = x.dtype();

        var q = self.q_proj.forward(x).splitAxis(.dout, .{ .h = self.num_heads, .hd = self.head_dim });
        var k = self.k_proj.forward(x).splitAxis(.dout, .{ .h = self.num_kv_heads, .hd = self.head_dim });
        var v = self.v_proj.forward(x).splitAxis(.dout, .{ .h = self.num_kv_heads, .hd = self.head_dim });

        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        q = q.convert(.f32);
        k = k.convert(.f32);
        v = v.convert(.f32);

        const cos, const sin = self.rotary_embed.getCosAndSin(position_ids, .f32);
        q = self.rotary_embed.applyRope(q, cos, sin);
        k = self.rotary_embed.applyRope(k, cos, sin);

        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });
        q = q.rename(.{ .s = .q });

        const new_kv_cache = kv_cache.update(k, v, token_index);

        const attn_output = zml.attention.attention.attention(
            q,
            new_kv_cache.keys().convert(.f32),
            new_kv_cache.values().convert(.f32),
            token_index,
            zml.attention.attention.Metadata.init(.fromBackend(.vanilla, x.dim(.s), self.num_heads)),
            zml.attention.attention.Parameters.init(.fromBackend(.vanilla)),
        ).merge(.{ .d = .{ .h, .hd } }).convert(orig_dtype);

        return .{ self.o_proj.forward(attn_output), new_kv_cache };
    }
};


pub const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{ .weight = store.createTensor("weight", .{.d}, .{ .d = .replicated }), .eps = eps };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    pub fn forward(self: RmsNorm, x: zml.Tensor) zml.Tensor {
        const normed = zml.nn.rmsNorm(x, .d, self.eps);
        return normed.mul(self.weight.broad(x.shape()));
    }
};

pub const TextRotaryEmbedding = struct {
    rope_opts: zml.nn.RopeOpts,
    rotary_dim: i64, // number of dims to rotate (only head_dim supported).
    mrope_section: [3]i64,

    pub fn init(head_dim: i64, rope_theta: f32, mrope_section: [3]i64) TextRotaryEmbedding {
        return .{
            .rope_opts = .{
                .layout = .sequential,
                .scaling = .{ .default = .{ .rope_theta = rope_theta } },
            },
            .rotary_dim = head_dim,
            .mrope_section = mrope_section,
        };
    }

    pub fn getCosAndSin(self: TextRotaryEmbedding, position_ids: zml.Tensor, dtype: zml.DataType) struct { zml.Tensor, zml.Tensor } {
        // position_ids: [.mrope=3, .b, .s]
        const s0 = self.mrope_section[0];
        const s1 = self.mrope_section[1];
        const s2 = self.mrope_section[2];
        // Tail supported on T only
        std.debug.assert(s0 >= s1 and s0 >= s2 and s1 == s2);

        const t_pos = position_ids.choose1d(.mrope, 0); // [b, s]
        const h_pos = position_ids.choose1d(.mrope, 1);
        const w_pos = position_ids.choose1d(.mrope, 2);

        const full_inv_freq = zml.nn.invFreq(self.rotary_dim, self.rope_opts).withTags(.{.hd});

        const freqs_t = t_pos.convert(.f32).outer(full_inv_freq);
        const freqs_h = h_pos.convert(.f32).outer(full_inv_freq);
        const freqs_w = w_pos.convert(.f32).outer(full_inv_freq);

        // Strided slices: pick every 3rd value from each modality's contribution
        const ft_i = freqs_t.slice1d(.hd, .{ .start = 0, .end = 3 * s1, .step = 3 }); // [b, s, s1]
        const fh_i = freqs_h.slice1d(.hd, .{ .start = 1, .end = 3 * s1, .step = 3 });
        const fw_i = freqs_w.slice1d(.hd, .{ .start = 2, .end = 3 * s2, .step = 3 });

        // Interleave: stack [b,s,n] × 3 → swap → merge → [b, s, 3n] = T H W T H W ...
        const interleaved = zml.Tensor.stack(&.{ ft_i, fh_i, fw_i }, .hd, .thw)
            .swapAxes(.thw, .hd)
            .merge(.{ .hd = .{ .hd, .thw } });

        const freqs = if (s0 > s1)
            // T tail: positions beyond H/W coverage stay pure T
            zml.Tensor.concatenate(&.{ interleaved, freqs_t.slice1d(.hd, .{ .start = 3 * s1 }) }, .hd)
        else
            interleaved;

        const emb = zml.Tensor.concatenate(&.{ freqs, freqs }, .hd);
        const cos = emb.cos().convert(dtype);
        const sin = emb.sin().convert(dtype);

        return .{ cos, sin };
    }

    fn rotateHalf(x: zml.Tensor) zml.Tensor {
        const half_dim = @divExact(x.dim(-1), 2);
        const x1 = x.slice1d(-1, .{ .start = 0, .end = half_dim });
        const x2 = x.slice1d(-1, .{ .start = half_dim, .end = x.dim(-1) });
        return zml.Tensor.concatenate(&.{ x2.negate(), x1 }, -1);
    }

    pub fn applyRope(self: TextRotaryEmbedding, x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const x_rot = x.slice1d(-1, .{ .start = 0, .end = self.rotary_dim });

        const cos_x = cos.insertAxes(.hd, .{.h}).broad(x_rot.shape());
        const sin_x = sin.insertAxes(.hd, .{.h}).broad(x_rot.shape());

        const rotated = x_rot.mul(cos_x).add(rotateHalf(x_rot).mul(sin_x));

        if (self.rotary_dim < x.dim(-1)) {
            const x_pass = x.slice1d(-1, .{ .start = self.rotary_dim, .end = x.dim(-1) });
            return zml.Tensor.concatenate(&.{ rotated, x_pass }, -1);
        } else {
            return rotated;
        }
    }
};

const Mlp = struct {
    gate_proj: zml.nn.Linear,
    up_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .gate_proj = .init(
                store.createTensor("gate_proj.weight", .{ .d_ffn, .d }, .{ .d_ffn = .model, .d = .replicated }),
                store.maybeCreateTensor("gate_proj.bias", .{ .d_ffn }, .{ .d_ffn = .model }),
                .d,
            ),
            .up_proj = .init(
                store.createTensor("up_proj.weight", .{ .d_ffn, .d }, .{ .d_ffn = .model, .d = .replicated }),
                store.maybeCreateTensor("up_proj.bias", .{ .d_ffn }, .{ .d_ffn = .model }),
                .d,
            ),
            .down_proj = .init(
                store.createTensor("down_proj.weight", .{ .d, .d_ffn }, .{ .d = .replicated, .d_ffn = .model }),
                store.maybeCreateTensor("down_proj.bias", .{ .d }, .{ .d = .replicated }),
                .d_ffn,
            ),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        self.up_proj.weight.deinit();
        if (self.up_proj.bias) |*bias| bias.deinit();
        self.gate_proj.weight.deinit();
        if (self.gate_proj.bias) |*bias| bias.deinit();
        self.down_proj.weight.deinit();
        if (self.down_proj.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const gated = self.gate_proj.forward(x).silu();
        const up_mul = self.up_proj.forward(x).mul(gated);
        return self.down_proj.forward(up_mul);
    }
};


pub const KvCache = struct {
    k: zml.Tensor,
    v: zml.Tensor,
    layer_index: zml.Tensor,

    pub fn init(config: TextConfig, max_seq_len: i64, dtype: zml.DataType) KvCache {
        const kv_shape = zml.Shape.init(.{ .layer = config.num_hidden_layers, .k = max_seq_len, .h = config.num_key_value_heads, .hd = config.head_dim }, dtype);
        return .{
            .k = .fromShape(kv_shape),
            .v = .fromShape(kv_shape),
            .layer_index = .init(.{}, .u32),
        };
    }

    pub fn initBuffer(kv: KvCache, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !zml.Bufferized(KvCache) {
        return .{
            .k = try zml.Buffer.uninitialized(io, platform, kv.k.shape(), sharding, .{}),
            .v = try zml.Buffer.uninitialized(io, platform, kv.v.shape(), sharding, .{}),
            .layer_index = try zml.Buffer.scalar(io, platform, 0, .u32),
        };
    }

    pub fn deinitBuffer(kv: *zml.Bufferized(KvCache)) void {
        kv.k.deinit();
        kv.v.deinit();
        kv.layer_index.deinit();
    }

    pub fn keys(kv: KvCache) zml.Tensor {
        return kv.k.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = kv.layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn values(kv: KvCache) zml.Tensor {
        return kv.v.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = kv.layer_index, .len = 1 } }).squeeze(.layer);
    }
    
    pub fn update(kv: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: zml.Tensor) KvCache {
        const k_shape = kv.k.shape().drop(.layer);
        return .{
            .k = kv.k.scatterSlices(
                .{ .layer = kv.layer_index, .k = token_index },
                new_k.convert(kv.k.dtype()).transpose(k_shape),
                .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
            ).reuseBuffer(kv.k),
            .v = kv.v.scatterSlices(
                .{ .layer = kv.layer_index, .k = token_index },
                new_v.convert(kv.v.dtype()).transpose(k_shape),
                .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
            ).reuseBuffer(kv.v),
            .layer_index = kv.layer_index,
        };
    }

    pub fn atLayer(kv: KvCache, layer_index: usize) KvCache {
        return .{
            .k = kv.k,
            .v = kv.v,
            .layer_index = zml.Tensor.scalar(layer_index, .u32),
        };
    }

    pub fn reuseBuffer(kv: KvCache, other: KvCache) KvCache {
        return .{
            .k = kv.k.reuseBuffer(other.k),
            .v = kv.v.reuseBuffer(other.v),
            .layer_index = kv.layer_index.reuseBuffer(other.layer_index),
        };
    }
};


pub const VisionModel = struct {
    patch_w: zml.Tensor, // [.vd, .c, .t, .kh, .kw]
    patch_bias: ?zml.Tensor, // [.vd]
    pos_embed: zml.Tensor, // [.pos=2304, .vd=1152] 2304 = 48×48 grid
    blocks: []VisionBlock,
    merger: VisionPatchMerger,
    // Slice (not [3]X) because zml.mem.bufferizeInner hits `unreachable` on .array.
    deepstack_mergers: []VisionPatchMerger,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: VisionConfig) !VisionModel {
        const blocks = try allocator.alloc(VisionBlock, @intCast(config.depth));
        errdefer allocator.free(blocks);

        const head_dim = @divExact(config.hidden_size, config.num_heads);
        for (blocks, 0..) |*block, i| {
            block.* = .init(store.withPrefix("blocks").withLayer(i), head_dim, config.rope_theta, config.norm_eps);
        }

        const deepstack_mergers = try allocator.alloc(VisionPatchMerger, 3);
        errdefer allocator.free(deepstack_mergers);
        deepstack_mergers[0] = .init(store.withPrefix("deepstack_merger_list").withLayer(0), config.norm_eps, true);
        deepstack_mergers[1] = .init(store.withPrefix("deepstack_merger_list").withLayer(1), config.norm_eps, true);
        deepstack_mergers[2] = .init(store.withPrefix("deepstack_merger_list").withLayer(2), config.norm_eps, true);

        return .{
            .patch_w = store.withPrefix("patch_embed.proj").createTensor(
                "weight", .{ .vd, .c, .t, .kh, .kw },
                .{ .vd = .replicated, .c = .replicated, .t = .replicated, .kh = .replicated, .kw = .replicated }),
            .patch_bias = store.withPrefix("patch_embed.proj").maybeCreateTensor("bias", .{.vd}, .{ .vd = .replicated }),
            .pos_embed = store.createTensor("pos_embed.weight", .{ .pos, .vd }, .{ .pos = .replicated, .vd = .replicated }),
            .blocks = blocks,
            .merger = .init(store.withPrefix("merger"), config.norm_eps, false),
            .deepstack_mergers = deepstack_mergers,
        };
    }

    pub fn deinit(self: VisionModel, allocator: std.mem.Allocator) void {
        allocator.free(self.blocks);
        allocator.free(self.deepstack_mergers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(VisionModel), allocator: std.mem.Allocator) void {
        self.patch_w.deinit();
        if (self.patch_bias) |*b| b.deinit();
        self.pos_embed.deinit();
        for (self.blocks) |*block| {
            VisionBlock.unloadBuffers(block);
        }
        allocator.free(self.blocks);
        VisionPatchMerger.unloadBuffers(&self.merger);
        for (self.deepstack_mergers) |*merger| {
            VisionPatchMerger.unloadBuffers(merger);
        }
        allocator.free(self.deepstack_mergers);
    }

    // Fixed-shape vision forward — compiled ONCE at (max_patches, pd).
    // grid_h/w are runtime tensors; attn_mask zeros padded-patch attention;
    // visual_mask zeros padded rows in the merger / deepstack outputs.
    pub fn paddedVisionForward(
        self: VisionModel,
        patches: zml.Tensor,        // [.p = max_patches, .pd]
        h_pos: zml.Tensor,          // [.p = max_patches] i64
        w_pos: zml.Tensor,          // [.p = max_patches] i64
        // additive attn bias: 0.0 for real patches, -1e9 for padded patches
        attn_mask: zml.Tensor,      // [.p = max_patches] f32
        // output mask: 1.0 for real visual tokens, 0.0 for padded
        visual_mask: zml.Tensor,    // [.n = max_n_visual] f32
        deepstack_indexes: [3]i64,
        grid_h: zml.Tensor,         // [] i64 — actual grid height (runtime)
        grid_w: zml.Tensor,         // [] i64 — actual grid width  (runtime)
    ) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
        const patches_typed = patches.convert(self.patch_w.dtype());
        const w_merged = self.patch_w.merge(.{ .pd = .{ .c, .t, .kh, .kw } });
        const patch_embd: zml.nn.Linear = .init(w_merged, self.patch_bias, .pd);
        var x = patch_embd.forward(patches_typed);  // [.p, .vd]

        // Bilinear pos_embed interpolation — same math as forward() but using
        // runtime grid_h/grid_w tensors instead of comptime i64 params.
        const pe_dim = self.pos_embed.dim(.pos);
        const pe_side: i64 = @intFromFloat(@sqrt(@as(f32, @floatFromInt(pe_dim))));
        const max_idx: i64 = pe_side - 1;
        const pe_dtype = self.pos_embed.dtype();

        const max_idx_f = zml.Tensor.scalar(@as(f32, @floatFromInt(max_idx)), .f32);
        const h_scale = max_idx_f.div(grid_h.convert(.f32).addConstant(-1.0));
        const w_scale = max_idx_f.div(grid_w.convert(.f32).addConstant(-1.0));
        const h_f = h_pos.convert(.f32).mul(h_scale);
        const w_f = w_pos.convert(.f32).mul(w_scale);

        const h0 = h_f.floor().convert(.i64);
        const w0 = w_f.floor().convert(.i64);
        const max_idx_t = zml.Tensor.scalar(max_idx, .i64);
        const h1 = h0.addConstant(1).minimum(max_idx_t);
        const w1 = w0.addConstant(1).minimum(max_idx_t);

        const wh = h_f.sub(h0.convert(.f32));
        const ww = w_f.sub(w0.convert(.f32));
        const one_minus_wh = wh.scale(@as(f32, -1)).addConstant(@as(f32, 1));
        const one_minus_ww = ww.scale(@as(f32, -1)).addConstant(@as(f32, 1));
        const c00 = one_minus_wh.mul(one_minus_ww);
        const c01 = one_minus_wh.mul(ww);
        const c10 = wh.mul(one_minus_ww);
        const c11 = wh.mul(ww);

        const idx00 = h0.scale(pe_side).add(w0);
        const idx01 = h0.scale(pe_side).add(w1);
        const idx10 = h1.scale(pe_side).add(w0);
        const idx11 = h1.scale(pe_side).add(w1);

        const p00 = self.pos_embed.gather(.{ .pos = idx00 }, .{}).convert(.f32);
        const p01 = self.pos_embed.gather(.{ .pos = idx01 }, .{}).convert(.f32);
        const p10 = self.pos_embed.gather(.{ .pos = idx10 }, .{}).convert(.f32);
        const p11 = self.pos_embed.gather(.{ .pos = idx11 }, .{}).convert(.f32);
        const target = p00.shape();
        const c00_b = c00.insertAxes(.last, .{.vd}).broad(target);
        const c01_b = c01.insertAxes(.last, .{.vd}).broad(target);
        const c10_b = c10.insertAxes(.last, .{.vd}).broad(target);
        const c11_b = c11.insertAxes(.last, .{.vd}).broad(target);
        const pos_f = p00.mul(c00_b).add(p01.mul(c01_b)).add(p10.mul(c10_b)).add(p11.mul(c11_b));
        const pos = pos_f.convert(pe_dtype);

        x = x.add(pos);

        // visual_mask comes in as [.n = max_n_visual] f32.
        // Rename .n → .p to match merger output tag, and convert dtype to match model.
        const vmask_p = visual_mask.rename(.{ .n = .p });  // [.p = max_n_visual] f32

        // Run transformer blocks with attention masking on padded patches.
        var deepstack_mergers_out: [3]zml.Tensor = undefined;
        for (self.blocks, 0..) |*block, i| {
            x = block.forward(x, h_pos, w_pos, attn_mask);
            for (deepstack_indexes, 0..) |layer_idx, idx| {
                if (i == @as(usize, @intCast(layer_idx))) {
                    const ds_raw = self.deepstack_mergers[idx].forward(x);  // [.p = max_n_visual, .d]
                    const ds_mask = vmask_p.convert(ds_raw.dtype()).insertAxes(.last, .{.d}).broad(ds_raw.shape());
                    deepstack_mergers_out[idx] = ds_raw.mul(ds_mask).rename(.{ .p = .n });
                }
            }
        }

        // Main merger output: zero padded rows, rename .p → .n for the prefill kernel.
        const visual_raw = self.merger.forward(x);  // [.p = max_n_visual, .d]
        const vis_mask = vmask_p.convert(visual_raw.dtype()).insertAxes(.last, .{.d}).broad(visual_raw.shape());
        const visual_tokens = visual_raw.mul(vis_mask).rename(.{ .p = .n });

        return .{ visual_tokens, deepstack_mergers_out[0], deepstack_mergers_out[1], deepstack_mergers_out[2] };
    }

    pub fn forward(
        self: VisionModel,
        patches: zml.Tensor,
        h_pos: zml.Tensor,
        w_pos: zml.Tensor,
        deepstack_indexes: [3]i64,
        grid_h: i64,
        grid_w: i64,
    ) struct { zml.Tensor, [3]zml.Tensor } {
        // Host patches arrive as f32; convert to weight dtype (bf16) for the matmul.
        // The 5-D conv kernel merges to a 2-D Linear weight here (graph op needs live MLIR ctx).
        const patches_typed = patches.convert(self.patch_w.dtype());
        const w_merged = self.patch_w.merge(.{ .pd = .{ .c, .t, .kh, .kw } });
        const patch_embd: zml.nn.Linear = .init(w_merged, self.patch_bias, .pd);
        const tokens = patch_embd.forward(patches_typed);

        // ── Bilinear interpolation of learned 48×48 pos_embed onto (grid_h, grid_w) ──
        // The model is trained to receive a bilinearly-resized pos_embed sized to the
        // actual image grid. A naive gather at h*48+w treats each patch as a literal
        // cell of the 48×48 canvas, which is wrong at any non-48 grid and collapses
        // the vision features. (Assumes grid_h, grid_w >= 2.)
        const pe_dim = self.pos_embed.dim(.pos);
        const pe_side: i64 = @intFromFloat(@sqrt(@as(f32, @floatFromInt(pe_dim))));
        const max_idx: i64 = pe_side - 1;
        const pe_dtype = self.pos_embed.dtype();

        // Map h_pos/w_pos from [0..grid-1] to continuous [0..47].
        const h_scale: f32 = @as(f32, @floatFromInt(max_idx)) / @as(f32, @floatFromInt(grid_h - 1));
        const w_scale: f32 = @as(f32, @floatFromInt(max_idx)) / @as(f32, @floatFromInt(grid_w - 1));
        const h_f = h_pos.convert(.f32).scale(h_scale);
        const w_f = w_pos.convert(.f32).scale(w_scale);

        const h0 = h_f.floor().convert(.i64);
        const w0 = w_f.floor().convert(.i64);
        // Clamp upper neighbor to 47 to avoid OOB at the bottom/right grid edge.
        const max_idx_t = zml.Tensor.scalar(max_idx, .i64);
        const h1 = h0.addConstant(1).minimum(max_idx_t);
        const w1 = w0.addConstant(1).minimum(max_idx_t);

        // Fractional offsets in [0, 1] and per-corner blend weights.
        const wh = h_f.sub(h0.convert(.f32));
        const ww = w_f.sub(w0.convert(.f32));
        const one_minus_wh = wh.scale(@as(f32, -1)).addConstant(@as(f32, 1));
        const one_minus_ww = ww.scale(@as(f32, -1)).addConstant(@as(f32, 1));
        const c00 = one_minus_wh.mul(one_minus_ww);
        const c01 = one_minus_wh.mul(ww);
        const c10 = wh.mul(one_minus_ww);
        const c11 = wh.mul(ww);

        // Flat indices into pos_embed [.pos = pe_side*pe_side].
        const idx00 = h0.scale(pe_side).add(w0);
        const idx01 = h0.scale(pe_side).add(w1);
        const idx10 = h1.scale(pe_side).add(w0);
        const idx11 = h1.scale(pe_side).add(w1);

        // Gather 4 corners, blend in f32 for precision, convert back.
        const p00 = self.pos_embed.gather(.{ .pos = idx00 }, .{}).convert(.f32);
        const p01 = self.pos_embed.gather(.{ .pos = idx01 }, .{}).convert(.f32);
        const p10 = self.pos_embed.gather(.{ .pos = idx10 }, .{}).convert(.f32);
        const p11 = self.pos_embed.gather(.{ .pos = idx11 }, .{}).convert(.f32);
        // Broadcast [.p] weights to [.p, .vd] for element-wise mul with the gathered corners.
        const target = p00.shape();
        const c00_b = c00.insertAxes(.last, .{.vd}).broad(target);
        const c01_b = c01.insertAxes(.last, .{.vd}).broad(target);
        const c10_b = c10.insertAxes(.last, .{.vd}).broad(target);
        const c11_b = c11.insertAxes(.last, .{.vd}).broad(target);
        const pos_f = p00.mul(c00_b).add(p01.mul(c01_b)).add(p10.mul(c10_b)).add(p11.mul(c11_b));
        const pos = pos_f.convert(pe_dtype);

        var x = tokens.add(pos);

        var deepstack_mergers_out: [3]zml.Tensor = undefined;

        for (self.blocks, 0..) |*block, i| {
            x = block.forward(x, h_pos, w_pos, null);
            for (deepstack_indexes, 0..) |layer_idx, idx| {
                if (i == @as(usize, @intCast(layer_idx))) {
                    deepstack_mergers_out[idx] = self.deepstack_mergers[idx].forward(x);
                }
            }
        }

        const visual_tokens = self.merger.forward(x);
        return .{ visual_tokens, deepstack_mergers_out };
    }
};

pub const VisionBlock = struct {
    norm1: zml.nn.LayerNorm,
    attn: VisionSelfAttn,
    norm2: zml.nn.LayerNorm,
    mlp: VisionMlp,

    pub fn init(store: zml.io.TensorStore.View, head_dim: i64, rope_theta: f32, norm_eps: f32) VisionBlock {
        return .{
            .norm1 = .{
                .weight = store.withPrefix("norm1").createTensor("weight", .{.vd}, .{ .vd = .replicated }),
                .bias = store.withPrefix("norm1").maybeCreateTensor("bias", .{.vd}, .{ .vd = .replicated }),
                .eps = norm_eps,
            },
            .attn = .init(store.withPrefix("attn"), head_dim, rope_theta),
            .norm2 = .{
                .weight = store.withPrefix("norm2").createTensor("weight", .{.vd}, .{ .vd = .replicated }),
                .bias = store.withPrefix("norm2").maybeCreateTensor("bias", .{.vd}, .{ .vd = .replicated }),
                .eps = norm_eps,
            },
            .mlp = .init(store.withPrefix("mlp")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(VisionBlock)) void {
        self.norm1.weight.deinit();
        if (self.norm1.bias) |*b| b.deinit();
        VisionSelfAttn.unloadBuffers(&self.attn);
        self.norm2.weight.deinit();
        if (self.norm2.bias) |*b| b.deinit();
        VisionMlp.unloadBuffers(&self.mlp);
    }

    pub fn forward(self: VisionBlock, x: zml.Tensor, h_pos: zml.Tensor, w_pos: zml.Tensor, attn_mask: ?zml.Tensor) zml.Tensor {
        const x1 = self.attn.forward(self.norm1.forward(x), h_pos, w_pos, attn_mask).add(x);
        return self.mlp.forward(self.norm2.forward(x1)).add(x1);
    }
};

pub const VisionPatchMerger = struct {
    norm: zml.nn.LayerNorm, // normalizes either d=1152 (vision hidden_size) or d=4608 (* spatial_merge_size^2 )
    fc1: zml.nn.Linear, // [4608 → 4608], with bias
    fc2: zml.nn.Linear, // [4608 → out_hidden_size], with bias
    use_postshuffle_norm: bool, // when true, DeepStack mergers, norm the 4608-dim merged vector after grouping

    pub fn init(store: zml.io.TensorStore.View, norm_eps: f32, comptime use_postshuffle_norm: bool) VisionPatchMerger {

        const norm_tags = if (use_postshuffle_norm) .{.d_merged} else .{.vd};
        const norm_sharding = if (use_postshuffle_norm) .{.d_merged = .replicated} else .{.vd = .replicated};
        
        return .{
            .norm = .{
                .weight = store.withPrefix("norm").createTensor("weight", norm_tags, norm_sharding),
                .bias = store.withPrefix("norm").maybeCreateTensor("bias", norm_tags, norm_sharding),
                .eps = norm_eps,
            },
            .fc1 = .init(
                store.withPrefix("linear_fc1").createTensor("weight", .{ .d_hidden, .d_merged }, .{ .d_hidden = .replicated, .d_merged = .replicated }),
                store.withPrefix("linear_fc1").maybeCreateTensor("bias", .{.d_hidden}, .{ .d_hidden = .replicated }),
                .d_merged,
            ),
            .fc2 = .init(
                store.withPrefix("linear_fc2").createTensor("weight", .{ .d, .d_hidden }, .{ .d_hidden = .replicated, .d = .replicated }),
                store.withPrefix("linear_fc2").maybeCreateTensor("bias", .{.d}, .{ .d = .replicated }),
                .d_hidden,
            ),
            .use_postshuffle_norm = use_postshuffle_norm,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(VisionPatchMerger)) void {
        self.norm.weight.deinit();
        if (self.norm.bias) |*b| b.deinit();
        self.fc1.weight.deinit();
        if (self.fc1.bias) |*b| b.deinit();
        self.fc2.weight.deinit();
        if (self.fc2.bias) |*b| b.deinit();
    }

    pub fn forward(self: VisionPatchMerger, patches: zml.Tensor) zml.Tensor {
        const p = @divExact(patches.dim(.p), 4);
        const d_merged = patches.dim(.vd) * 4;

        const normed_merged = if (self.use_postshuffle_norm)
            self.norm.forward(patches.reshape(.{ .p = p, .d_merged = d_merged }))
        else
            self.norm.forward(patches).reshape(.{ .p = p, .d_merged = d_merged });

        
        const hidden = self.fc1.forward(normed_merged).gelu();
        return self.fc2.forward(hidden);
    }
};

pub const VisionMlp = struct {
    fc1: zml.nn.Linear, // [vd, vd_ff]
    fc2: zml.nn.Linear, // [vd_ff, vd]

    pub fn init(store: zml.io.TensorStore.View) VisionMlp {
        return .{
            .fc1 = .init(
                store.withPrefix("linear_fc1").createTensor("weight", .{ .vd_ff, .vd }, .{ .vd = .replicated, .vd_ff = .replicated }),
                store.withPrefix("linear_fc1").maybeCreateTensor("bias", .{.vd_ff}, .{ .vd_ff = .replicated }),
                .vd,
            ),
            .fc2 = .init(
                store.withPrefix("linear_fc2").createTensor("weight", .{ .vd, .vd_ff }, .{ .vd = .replicated, .vd_ff = .replicated }),
                store.withPrefix("linear_fc2").maybeCreateTensor("bias", .{.vd}, .{ .vd = .replicated }),
                .vd_ff,
            ),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(VisionMlp)) void {
        self.fc1.weight.deinit();
        if (self.fc1.bias) |*b| b.deinit();
        self.fc2.weight.deinit();
        if (self.fc2.bias) |*b| b.deinit();
    }

    pub fn forward(self: VisionMlp, hidden_state: zml.Tensor) zml.Tensor {
        var x = self.fc1.forward(hidden_state);
        x = x.gelu();
        return self.fc2.forward(x);
    }
};


pub const VisionRotaryEmbedding = struct {
    half_rotary_dim: i64, // head_dim / 2  (e.g. 36 for head_dim=72)
    theta: f32,

    pub fn init(head_dim: i64, theta: f32) VisionRotaryEmbedding {
        return .{ .half_rotary_dim = @divExact(head_dim, 2), .theta = theta };
    }

    // h_pos, w_pos: [.s] integer grid coordinates for each patch.
    // Returns cos, sin each shaped [.p, .hd=head_dim].
    pub fn getCosAndSin(
        self: VisionRotaryEmbedding,
        h_pos: zml.Tensor,
        w_pos: zml.Tensor,
    ) struct { zml.Tensor, zml.Tensor } {
        const rope_opts: zml.nn.RopeOpts = .{
            .layout = .sequential,
            .scaling = .{ .default = .{ .rope_theta = self.theta } },
        };
        // inv_freq: [.hd = half_rotary_dim/2] (e.g. 18 entries for head_dim=72)
        const inv_freq = zml.nn.invFreq(self.half_rotary_dim, rope_opts).withTags(.{.hd});
        const freqs_h = h_pos.convert(.f32).outer(inv_freq); // [.p, .hd=18]
        const freqs_w = w_pos.convert(.f32).outer(inv_freq); // [.p, .hd=18]
        const emb = zml.Tensor.concatenate(&.{ freqs_h, freqs_w }, .hd); // [.p, .hd=36]
        const doubled = zml.Tensor.concatenate(&.{ emb, emb }, .hd); // [.p, .hd=72]
        // Keep cos/sin in f32: HF applies vision RoPE in float32 regardless of model dtype.
        return .{ doubled.cos(), doubled.sin() };
    }

    fn rotateHalf(x: zml.Tensor) zml.Tensor {
        const half_dim = @divExact(x.dim(.hd), 2);
        const x1 = x.slice1d(.hd, .{ .start = 0, .end = half_dim });
        const x2 = x.slice1d(.hd, .{ .start = half_dim, .end = x.dim(.hd) });
        return zml.Tensor.concatenate(&.{ x2.negate(), x1 }, .hd);
    }

    // x: [.p, .h, .hd], cos/sin: [.p, .hd] in f32.
    // Matches HF apply_rotary_pos_emb_vision: upcast x to f32, rotate, convert back.
    pub fn applyVisionRope(x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const orig_dtype = x.dtype();
        const xf = x.convert(.f32);
        const cos_x = cos.insertAxes(.hd, .{.h}).broad(xf.shape());
        const sin_x = sin.insertAxes(.hd, .{.h}).broad(xf.shape());
        return xf.mul(cos_x).add(rotateHalf(xf).mul(sin_x)).convert(orig_dtype);
    }
};

pub const VisionSelfAttn = struct {
    qkv_proj: zml.nn.Linear,
    proj: zml.nn.Linear,
    vision_rotary_embd: VisionRotaryEmbedding,
    head_dim: i64,

    pub fn init(store: zml.io.TensorStore.View, head_dim: i64, rope_theta: f32) VisionSelfAttn {
        return .{ 
            .qkv_proj = .init(store.createTensor("qkv.weight", .{ .qkv_out, .vd }, .{ .qkv_out = .replicated, .vd = .replicated }), store.createTensor("qkv.bias", .{ .qkv_out }, .{ .qkv_out = .replicated }), .vd),
            .proj = .init(store.createTensor("proj.weight", .{ .vd, .dout }, .{ .vd = .replicated, .dout = .replicated }), store.createTensor("proj.bias", .{ .vd }, .{ .vd = .replicated }), .dout),
            .vision_rotary_embd = .init(head_dim, rope_theta),
            .head_dim = head_dim
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(VisionSelfAttn)) void {
        self.qkv_proj.weight.deinit();
        if (self.qkv_proj.bias) |*bias| bias.deinit();
        self.proj.weight.deinit();
        if (self.proj.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: VisionSelfAttn, x: zml.Tensor, h_pos: zml.Tensor, w_pos: zml.Tensor, attn_mask: ?zml.Tensor) zml.Tensor {
        const qkv = self.qkv_proj.forward(x);
        const hd = self.head_dim;
        const h = @divExact(qkv.dim(.qkv_out), 3*hd); // num heads

        var q, var k, var v = qkv.chunkExact(.qkv_out, 3);
        q = q.splitAxis(.qkv_out, .{ .h = h, .hd = hd });
        k = k.splitAxis(.qkv_out, .{ .h = h, .hd = hd });
        v = v.splitAxis(.qkv_out, .{ .h = h, .hd = hd });

        const cos, const sin = self.vision_rotary_embd.getCosAndSin(h_pos, w_pos);

        q = VisionRotaryEmbedding.applyVisionRope(q, cos, sin);
        k = VisionRotaryEmbedding.applyVisionRope(k, cos, sin);

        // rename dims for attention
        k = k.rename(.{ .p = .k });
        v = v.rename(.{ .p = .k });
        q = q.rename(.{ .p = .q });

        // attn_mask: [.k = max_p] additive bias — 0.0 for real patches, -1e9 for padded.
        // Convert to match attention-weight dtype (bf16 for model weights) before sdpa adds it.
        const sdpa_opts: zml.nn.SdpaOpts = if (attn_mask) |m| .{
            .attn_mask = m.withTags(.{.k}).convert(k.dtype()),
        } else .{};
        const attn_output = zml.nn.sdpa(q, k, v, sdpa_opts).merge(.{ .dout = .{ .h, .hd } }).rename(.{.q = .p});

        return self.proj.forward(attn_output);
    }
};
