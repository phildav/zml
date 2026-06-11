const std = @import("std");
const zigimg = @import("zigimg");

const norm_mean: [3]f32 = .{ 0.5, 0.5, 0.5 };
const norm_std: [3]f32 = .{ 0.5, 0.5, 0.5 };

pub const Preprocessed = struct {
    patches: []f32, // [actual_num_patches * patch_embed_dim]
    h_pos: []i64,
    w_pos: []i64,
    grid_h: i64,
    grid_w: i64,

    pub fn deinit(self: Preprocessed, allocator: std.mem.Allocator) void {
        allocator.free(self.patches);
        allocator.free(self.h_pos);
        allocator.free(self.w_pos);
    }

    pub fn numPatches(self: Preprocessed) i64 {
        return self.grid_h * self.grid_w;
    }
};

/// Load and preprocess an image for Qwen3-VL.
/// Returns patches in [num_patches, patch_embed_dim] layout where the inner
/// dimension is ordered [in_channels, temporal_patch_size, patch_h, patch_w] —
/// matching the vision encoder weight shape [vd, in_ch, tps, ph, pw].
/// Both temporal frames are identical (single-image duplication for video model).
pub fn preprocess(
    allocator: std.mem.Allocator,
    image_bytes: []const u8,
    target_num_patches: i64,
    patch_size: i64,
    temporal_patch_size: i64,
    in_channels: i64,
    spatial_merge_size: i64,
) !Preprocessed {
    var img = try zigimg.Image.fromMemory(allocator, image_bytes);
    defer img.deinit(allocator);
    try img.convert(allocator, .rgb24);

    const src_w: i64 = @intCast(img.width);
    const src_h: i64 = @intCast(img.height);

    const grid_h, const grid_w = computeGrid(src_h, src_w, target_num_patches, patch_size, spatial_merge_size);

    const dst_h: usize = @intCast(grid_h * patch_size);
    const dst_w: usize = @intCast(grid_w * patch_size);

    const resized = try bicubicResize(
        allocator,
        img.pixels.rgb24,
        @intCast(src_w),
        @intCast(src_h),
        dst_w,
        dst_h,
    );
    defer allocator.free(resized);

    const num_patches: usize = @intCast(grid_h * grid_w);
    const ps: usize = @intCast(patch_size);
    const tps: usize = @intCast(temporal_patch_size);
    const ic: usize = @intCast(in_channels);
    const patch_embed_dim = ic * tps * ps * ps;

    const patches = try allocator.alloc(f32, num_patches * patch_embed_dim);
    errdefer allocator.free(patches);

    extractPatches(patches, resized, dst_w, @intCast(grid_h), @intCast(grid_w), ps, tps, ic, @intCast(spatial_merge_size));

    const grid_h_merged: usize = @intCast(@divExact(grid_h, spatial_merge_size));
    const grid_w_merged: usize = @intCast(@divExact(grid_w, spatial_merge_size));
    const h_pos = try allocator.alloc(i64, num_patches);
    errdefer allocator.free(h_pos);
    const w_pos = try allocator.alloc(i64, num_patches);
    errdefer allocator.free(w_pos);

    var idx: usize = 0;
    const sm_u: usize = @intCast(spatial_merge_size);
    // Iteration order must match extractPatches: patch[i] and h_pos[i]/w_pos[i] index the same patch.
    for (0..grid_h_merged) |mh| {
        for (0..grid_w_merged) |mw| {
            for (0..sm_u) |sh| {
                for (0..sm_u) |sw| {
                    h_pos[idx] = @intCast(mh * sm_u + sh);
                    w_pos[idx] = @intCast(mw * sm_u + sw);
                    idx += 1;
                }
            }
        }
    }

    return .{ .patches = patches, .h_pos = h_pos, .w_pos = w_pos, .grid_h = grid_h, .grid_w = grid_w };
}

fn computeGrid(
    img_h: i64,
    img_w: i64,
    target: i64,
    patch_size: i64,
    merge_size: i64,
) struct { i64, i64 } {
    const step = patch_size * merge_size;
    // Natural grid: round image to nearest multiple of step (banker's rounding, matches HF smart_resize)
    const nat_gh = @max(merge_size, @divExact(roundToMultiple(img_h, step), patch_size));
    const nat_gw = @max(merge_size, @divExact(roundToMultiple(img_w, step), patch_size));

    if (nat_gh * nat_gw <= target) return .{ nat_gh, nat_gw };

    // Scale down to fit target, preserving aspect ratio:
    //   gh * gw = target  and  gw/gh = aspect  →  gh = sqrt(target / aspect)
    const aspect = @as(f64, @floatFromInt(img_w)) / @as(f64, @floatFromInt(img_h));
    const gh_f = @sqrt(@as(f64, @floatFromInt(target)) / aspect);
    const gw_f = gh_f * aspect;

    var gh = @max(merge_size, alignDown(@as(i64, @intFromFloat(gh_f)), merge_size));
    var gw = @max(merge_size, alignDown(@as(i64, @intFromFloat(gw_f)), merge_size));

    // Nudge down one merge_size step at a time until we fit
    while (gh * gw > target) {
        if (gw > gh) gw -= merge_size else gh -= merge_size;
        if (gh < merge_size or gw < merge_size) break;
    }

    return .{ @max(merge_size, gh), @max(merge_size, gw) };
}

fn roundToMultiple(v: i64, to: i64) i64 {
    const half = @divExact(to, 2);
    const rem = @mod(v, to);
    const base = v - rem;
    if (rem > half) return base + to;
    if (rem < half) return base;
    // Tie: round to even (base/to must be even)
    const n = @divExact(base, to);
    return if (@mod(n, 2) == 0) base else base + to;
}

fn alignUp(v: i64, to: i64) i64 {
    return @divFloor(v + to - 1, to) * to;
}

fn alignDown(v: i64, to: i64) i64 {
    return @max(to, @divFloor(v, to) * to);
}

const Rgb24 = zigimg.color.Rgb24;

/// Catmull-Rom cubic kernel (a=-0.5), matching Pillow's BICUBIC resampling used by HF.
inline fn cubicWeight(t: f64) f64 {
    const a = -0.5;
    const abs_t = @abs(t);
    if (abs_t <= 1.0) return (a + 2.0) * abs_t * abs_t * abs_t - (a + 3.0) * abs_t * abs_t + 1.0;
    if (abs_t < 2.0) return a * abs_t * abs_t * abs_t - 5.0 * a * abs_t * abs_t + 8.0 * a * abs_t - 4.0 * a;
    return 0.0;
}

fn bicubicResize(
    allocator: std.mem.Allocator,
    src: []const Rgb24,
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) ![]Rgb24 {
    const dst = try allocator.alloc(Rgb24, dst_w * dst_h);
    errdefer allocator.free(dst);

    const sx_scale = if (dst_w > 1) @as(f64, @floatFromInt(src_w)) / @as(f64, @floatFromInt(dst_w)) else 1.0;
    const sy_scale = if (dst_h > 1) @as(f64, @floatFromInt(src_h)) / @as(f64, @floatFromInt(dst_h)) else 1.0;

    for (0..dst_h) |dy| {
        for (0..dst_w) |dx| {
            // Map dst pixel center to src coordinates (PIL/Pillow convention)
            const sx_f = ((@as(f64, @floatFromInt(dx)) + 0.5) * sx_scale) - 0.5;
            const sy_f = ((@as(f64, @floatFromInt(dy)) + 0.5) * sy_scale) - 0.5;

            const x_int: i64 = @intFromFloat(@floor(sx_f));
            const y_int: i64 = @intFromFloat(@floor(sy_f));
            const fx = sx_f - @as(f64, @floatFromInt(x_int));
            const fy = sy_f - @as(f64, @floatFromInt(y_int));

            var r: f64 = 0.0;
            var g: f64 = 0.0;
            var b: f64 = 0.0;

            var m: i64 = -1;
            while (m <= 2) : (m += 1) {
                const wy = cubicWeight(@as(f64, @floatFromInt(m)) - fy);
                if (wy == 0.0) continue;
                const sy: usize = @intCast(@min(@max(y_int + m, 0), @as(i64, @intCast(src_h - 1))));
                var n: i64 = -1;
                while (n <= 2) : (n += 1) {
                    const wx = cubicWeight(@as(f64, @floatFromInt(n)) - fx);
                    if (wx == 0.0) continue;
                    const sx: usize = @intCast(@min(@max(x_int + n, 0), @as(i64, @intCast(src_w - 1))));
                    const p = src[sy * src_w + sx];
                    const w = wx * wy;
                    r += w * @as(f64, @floatFromInt(p.r));
                    g += w * @as(f64, @floatFromInt(p.g));
                    b += w * @as(f64, @floatFromInt(p.b));
                }
            }

            dst[dy * dst_w + dx] = .{
                .r = @intFromFloat(@min(255.0, @max(0.0, @round(r)))),
                .g = @intFromFloat(@min(255.0, @max(0.0, @round(g)))),
                .b = @intFromFloat(@min(255.0, @max(0.0, @round(b)))),
            };
        }
    }
    return dst;
}

// Patch extraction + normalization
/// Fill `out` [num_patches, patch_embed_dim] with normalized patch values.
/// Inner dimension ordering: [in_channels, temporal_frames, patch_h, patch_w].
/// Outer patch order is **merge-grouped**: 4 consecutive .p entries form one 2×2
/// spatial block, matching what `Qwen3VLVisionPatchMerger` expects when it
/// reshapes [.p = p/(sm*sm), .d_merged = d*sm*sm].
/// Both temporal frames receive the same pixel values (single-image duplication).
fn extractPatches(
    out: []f32,
    pixels: []const Rgb24,
    img_w: usize,
    grid_h: usize,
    grid_w: usize,
    ps: usize,
    tps: usize,
    ic: usize,
    sm: usize,
) void {
    const ped = ic * tps * ps * ps;
    const merged_h = grid_h / sm;
    const merged_w = grid_w / sm;
    var p_idx: usize = 0;
    for (0..merged_h) |mh| {
        for (0..merged_w) |mw| {
            for (0..sm) |sh| {
                for (0..sm) |sw| {
                    const gh = mh * sm + sh;
                    const gw = mw * sm + sw;
                    const out_patch = out[p_idx * ped ..][0..ped];
                    p_idx += 1;
                    for (0..ic) |c| {
                        for (0..ps) |py| {
                            for (0..ps) |px| {
                                const pixel = pixels[(gh * ps + py) * img_w + (gw * ps + px)];
                                const raw: f32 = @floatFromInt(switch (c) {
                                    0 => pixel.r,
                                    1 => pixel.g,
                                    else => pixel.b,
                                });
                                const val = (raw / 255.0 - norm_mean[c]) / norm_std[c];
                                for (0..tps) |t| {
                                    out_patch[c * (tps * ps * ps) + t * (ps * ps) + py * ps + px] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
