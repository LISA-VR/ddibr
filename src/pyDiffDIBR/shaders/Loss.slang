/*
 This file is part of "∂DIBR", a differentiable renderer based on Depth Image Based Rendering (DIBR) techniques for fast Novel View Synthesis.
 Copyright (C) 2026 "Université Libre de Bruxelles (ULB)". All rights reserved.
 
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as published
 by the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.
 
 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
 
 Contact:
     - Armand Losfeld - armand.lfd.pro@proton.me
     - Daniele Bonatto - daniele.bonatto@ulb.be
*/

import DiffTex;
import _Utils;
import _Constants;

/**
 *   -------------------------------------------------------------------------------------------------
 *   L1 NORM
 *   -------------------------------------------------------------------------------------------------
 */

[BackwardDifferentiable]
vector<float, N> l1_norm<let N : int>(
    no_diff CstTexture<N> target,
    DifferentiableTexture<N> input,
    no_diff int2 loc,
    no_diff int w,
    no_diff int h,
) {
    vector<float, N> res;

    vector<float, N> c_t = target.LoadTexel(loc, w, h);
    vector<float, N> c_i = input.LoadTexel(loc, w, h);

    res = (c_t - c_i);

    return abs(res);
};

[shader("compute")]
[NumThreads(32, 32, 1)]
void main_bwd_l1_norm_color(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform CstTexture<4> target,
    uniform DifferentiableTexture<4> input
) {
    int width, height;
    input.GetDimensions(width, height);
    if ((threadIdx.x < 0) || (threadIdx.x >= width) || (threadIdx.y < 0) || (threadIdx.y >= height)) { return;}

    bwd_diff(l1_norm)(target, input, threadIdx.xy, width, height, float4(1.f));
};

[shader("compute")]
[NumThreads(32, 32, 1)]
void main_fwd_bwd_l1_norm_color(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform CstTexture<4> target,
    uniform DifferentiableTexture<4> input,
    uniform DifferentiableTexture<4> loss,
    uniform float scale_loss,
) {
    int width, height;
    input.GetDimensions(width, height);
    if ((threadIdx.x < 0) || (threadIdx.x >= width) || (threadIdx.y < 0) || (threadIdx.y >= height)) { return;}

    float4 l = l1_norm<4>(target, input, threadIdx.xy, width, height);

    loss.StoreTexel(
        threadIdx.xy,
        scale_loss * l
    );

    bwd_diff(l1_norm<4>)(target, input, threadIdx.xy, width, height, float4(scale_loss));
};

[shader("compute")]
[NumThreads(32, 32, 1)]
void main_fwd_bwd_l1_norm_cst_diff_depth(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform CstTexture<1> target,
    uniform DifferentiableTexture<1> input,
    uniform DifferentiableTexture<4> loss,
    uniform float scale_loss,
) {
    int width, height;
    input.GetDimensions(width, height);
    if ((threadIdx.x < 0) || (threadIdx.x >= width) || (threadIdx.y < 0) || (threadIdx.y >= height)) { return;}

    float4 l = l1_norm<1>(target, input, threadIdx.xy, width, height);

    loss.StoreTexel(
        threadIdx.xy,
        scale_loss * l
    );

    bwd_diff(l1_norm<1>)(target, input, threadIdx.xy, width, height, scale_loss);
};

[BackwardDifferentiable]
vector<float, N> l1_norm_2<let N : int>(
    DifferentiableTexture<N> target,
    DifferentiableTexture<N> input,
    no_diff int2 loc,
    no_diff int w,
    no_diff int h,
) {
    vector<float, N> res;

    vector<float, N> c_t = target.LoadTexel(loc, w, h);
    vector<float, N> c_i = input.LoadTexel(loc, w, h);

    res = (c_t - c_i);

    return abs(res);
};

[shader("compute")]
[NumThreads(32, 32, 1)]
void main_fwd_bwd_l1_norm_diff_diff_depth(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform DifferentiableTexture<1> target,
    uniform DifferentiableTexture<1> input,
    uniform DifferentiableTexture<4> loss,
    uniform float scale_loss,
) {
    int width, height;
    input.GetDimensions(width, height);
    if ((threadIdx.x < 0) || (threadIdx.x >= width) || (threadIdx.y < 0) || (threadIdx.y >= height)) { return;}

    float4 l = l1_norm_2<1>(target, input, threadIdx.xy, width, height);

    loss.StoreTexel(
        threadIdx.xy,
        scale_loss * l
    );

    bwd_diff(l1_norm_2<1>)(target, input, threadIdx.xy, width, height, scale_loss);
};

[BackwardDifferentiable]
vector<float, N> mask_l1_norm<let N : int>(
    CstTexture<N> target,
    DifferentiableTexture<N> input,
    DifferentiableTexture<1> mask,
    no_diff int2 loc,
    no_diff int w,
    no_diff int h,
) {
    vector<float, N> res;

    vector<float, N> c_t = target.LoadTexel(loc, w, h);
    vector<float, N> c_i = input.LoadTexel(loc, w, h);
    vector<float, 1> m   = mask.LoadTexel(loc, w, h);

    res = (c_t - c_i);

    return m.x * abs(res);
};

[BackwardDifferentiable]
vector<float, N> mask_l1_norm_2<let N : int>(
    DifferentiableTexture<N> target,
    DifferentiableTexture<N> input,
    DifferentiableTexture<1> mask,
    no_diff int2 loc,
    no_diff int w,
    no_diff int h,
) {
    vector<float, N> res;

    vector<float, N> c_t = target.LoadTexel(loc, w, h);
    vector<float, N> c_i = input.LoadTexel(loc, w, h);
    vector<float, 1> m   = mask.LoadTexel(loc, w, h);

    res = (c_t - c_i);

    return m.x * abs(res);
};

[shader("compute")]
[NumThreads(32, 32, 1)]
void main_fwd_bwd_masked_l1_norm_diff_diff_depth(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform DifferentiableTexture<1> target,
    uniform DifferentiableTexture<1> input,
    uniform DifferentiableTexture<1> mask,
    uniform DifferentiableTexture<4> loss,
    uniform float scale_loss,
) {
    int width, height;
    input.GetDimensions(width, height);
    if ((threadIdx.x < 0) || (threadIdx.x >= width) || (threadIdx.y < 0) || (threadIdx.y >= height)) { return;}

    float4 l = mask_l1_norm_2<1>(target, input, mask, threadIdx.xy, width, height);

    loss.StoreTexel(
        threadIdx.xy,
        scale_loss * l
    );

    bwd_diff(mask_l1_norm_2<1>)(target, input, mask, threadIdx.xy, width, height, scale_loss);
};

[shader("compute")]
[NumThreads(32, 32, 1)]
void main_fwd_bwd_masked_l1_norm_color(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform CstTexture<4> target,
    uniform DifferentiableTexture<4> input,
    uniform DifferentiableTexture<1> mask,
    uniform DifferentiableTexture<4> loss,
    uniform float scale_loss,
) {
    int width, height;
    input.GetDimensions(width, height);
    if ((threadIdx.x < 0) || (threadIdx.x >= width) || (threadIdx.y < 0) || (threadIdx.y >= height)) { return;}

    float4 l = mask_l1_norm<4>(target, input, mask, threadIdx.xy, width, height);

    loss.StoreTexel(
        threadIdx.xy,
        scale_loss * l
    );

    bwd_diff(mask_l1_norm<4>)(target, input, mask, threadIdx.xy, width, height, scale_loss);
};

/**
 *   -------------------------------------------------------------------------------------------------
 *   L2 NORM
 *   -------------------------------------------------------------------------------------------------
*/

[BackwardDifferentiable]
vector<float, N> l2_norm<let N : int>(
    no_diff CstTexture<N> target,
    DifferentiableTexture<N> input,
    no_diff int2 loc,
    no_diff int w,
    no_diff int h,
) {
    vector<float, N> res;

    vector<float, N> c_t = target.LoadTexel(loc, w, h);
    vector<float, N> c_i = input.LoadTexel(loc, w, h);

    res = (c_t - c_i);

    return res * res;
};

[BackwardDifferentiable]
vector<float, N> masked_l2_norm<let N : int>(
    CstTexture<N> target,
    DifferentiableTexture<N> input,
    DifferentiableTexture<1> mask,
    no_diff int2 loc,
    no_diff int w,
    no_diff int h,
) {
    vector<float, N> res;

    vector<float, N> c_t = target.LoadTexel(loc, w, h);
    vector<float, N> c_i = input.LoadTexel(loc, w, h);
    vector<float, 1> m   = mask.LoadTexel(loc, w, h);

    res = (c_t - c_i);

    return m.x * (res * res);
};

[shader("compute")]
[NumThreads(32, 32, 1)]
void main_bwd_l2_norm_color(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform CstTexture<4> target,
    uniform DifferentiableTexture<4> input
) {
    int width, height;
    input.GetDimensions(width, height);
    if ((threadIdx.x < 0) || (threadIdx.x >= width) || (threadIdx.y < 0) || (threadIdx.y >= height)) { return;}

    bwd_diff(l2_norm)(target, input, threadIdx.xy, width, height, float4(1.f, 1.f, 1.f, 0.f));
};

[shader("compute")]
[NumThreads(32, 32, 1)]
void main_call_main_bwd_l2_norm_color(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform CstTexture<4> target,
    uniform DifferentiableTexture<4> input,
    uniform DifferentiableTexture<4> loss,
    uniform float scale_loss,
) {
    int width, height;
    input.GetDimensions(width, height);
    if ((threadIdx.x < 0) || (threadIdx.x >= width) || (threadIdx.y < 0) || (threadIdx.y >= height)) { return;}

    float4 l2_ = l2_norm<4>(target, input, threadIdx.xy, width, height);

    loss.StoreTexel(
        threadIdx.xy,
        scale_loss * l2_
    );

    bwd_diff(l2_norm<4>)(target, input, threadIdx.xy, width, height, float4(scale_loss));
};

[shader("compute")]
[NumThreads(32, 32, 1)]
void main_call_main_bwd_masked_l2_norm_color(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform CstTexture<4> target,
    uniform DifferentiableTexture<4> input,
    uniform DifferentiableTexture<1> mask,
    uniform DifferentiableTexture<4> loss,
    uniform float scale_loss,
) {
    int width, height;
    input.GetDimensions(width, height);
    if ((threadIdx.x < 0) || (threadIdx.x >= width) || (threadIdx.y < 0) || (threadIdx.y >= height)) { return;}

    float4 l2_ = masked_l2_norm<4>(target, input, mask, threadIdx.xy, width, height);

    loss.StoreTexel(
        threadIdx.xy,
        scale_loss * l2_
    );

    bwd_diff(masked_l2_norm<4>)(target, input, mask, threadIdx.xy, width, height, float4(scale_loss));
};

/**
 *   -------------------------------------------------------------------------------------------------
 *   Anisotropic Total Variation
 *   -------------------------------------------------------------------------------------------------
*/

[BackwardDifferentiable]
vector<float, N> anisotropic_total_variation<let N : int>(
    DifferentiableTexture<N> texture,
    no_diff int2 loc,
    no_diff int w,
    no_diff int h
) {
    if ((loc.x > w - 1) || (loc.y > h - 1)) { return vector < float, N >(0.f);} 

    vector<float, N> val_xy      = texture.LoadTexel(loc, w, h);
    vector<float, N> variation_x = texture.LoadTexel(loc + int2(1, 0), w, h) - val_xy;
    vector<float, N> variation_y = texture.LoadTexel(loc + int2(0, 1), w, h) - val_xy;

    return abs(variation_x) + abs(variation_y);
};

[BackwardDifferentiable]
vector<float, N> anisotropic_total_variation_w_sigmoid<let N : int>(
    DifferentiableTexture<N> texture,
    no_diff int2 loc,
    no_diff int w,
    no_diff int h,
    no_diff float shift,
    no_diff float scale,
) {
    //if ((loc.x > w - 1) || (loc.y > h - 1)) { return vector<float, N>(0.f);}

    vector<float, N> val_xy      = sigmoid_shift_scale(texture.LoadTexel(loc, w, h)             , 1.0f, -1.0f, 2.0f); // put depth values between [-1, 1]
    vector<float, N> variation_x = sigmoid_shift_scale(texture.LoadTexel(loc + int2(1, 0), w, h), 1.0f, -1.0f, 2.0f) - val_xy;
    vector<float, N> variation_y = sigmoid_shift_scale(texture.LoadTexel(loc + int2(0, 1), w, h), 1.0f, -1.0f, 2.0f) - val_xy;

    return abs(variation_x) + abs(variation_y);
};

[shader("compute")]
[NumThreads(32, 32, 1)]
void main_bwd_atv_depth(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform DifferentiableTexture<1> input
) {
    int width, height;
    input.GetDimensions(width, height);
    if ((threadIdx.x < 0) || (threadIdx.x >= width) || (threadIdx.y < 0) || (threadIdx.y >= height)) { return;}

    bwd_diff(anisotropic_total_variation)(input, threadIdx.xy, width, height, float1(1.f));
}

[shader("compute")]
[NumThreads(32, 32, 1)]
void main_call_main_bwd_atv_depth(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform DifferentiableTexture<1> input,
    uniform DifferentiableTexture<4> loss,
    uniform float scale_loss,
    uniform float shift_sigmoid,
    uniform float scale_sigmoid,
) {
    int width, height;
    input.GetDimensions(width, height);
    if ((threadIdx.x < 0) || (threadIdx.x >= width) || (threadIdx.y < 0) || (threadIdx.y >= height)) { return;}

    float tv_ = anisotropic_total_variation_w_sigmoid<1>(input, threadIdx.xy, width, height, shift_sigmoid, scale_sigmoid).x;
    
    loss.StoreTexel(
        threadIdx.xy,
        scale_loss * float4(tv_) 
    );

    // bwd_diff(anisotropic_total_variation<1>)(input, threadIdx.xy, width, height, float1(1.f));
    bwd_diff(anisotropic_total_variation_w_sigmoid<1>)(input, threadIdx.xy, width, height, shift_sigmoid, scale_sigmoid, float1(scale_loss));
};

/**
 *   -------------------------------------------------------------------------------------------------
 *   Depth Consistency Loss
 *   -------------------------------------------------------------------------------------------------
*/

[BackwardDifferentiable]
float depth_consistency_loss(
    DifferentiableTexture<1> rendered_depth_per_input,
    DifferentiableTexture<1> blended_depth,
    DifferentiableTexture<1> rendered_quality_per_input,
    DifferentiableTexture<1> blended_quality,
    no_diff int2 loc,
    no_diff int w,
    no_diff int h,
) {
    float res;

    float d_i   = rendered_depth_per_input.LoadTexel(loc, w, h).x;
    float d     = blended_depth.LoadTexel(loc, w, h).x;
    float q_i   = rendered_quality_per_input.LoadTexel(loc, w, h).x;
    float q     = blended_quality.LoadTexel(loc, w, h).x;
    if (q < cst_type_value<float>.EPS()){return 0.f;}

    res = q_i / q * abs(d - d_i);

    return res;
};

[shader("compute")]
[NumThreads(32, 32, 1)]
void main_fwd_bwd_depth_consistency_loss(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform DifferentiableTexture<1> rendered_depth_per_input,
    uniform DifferentiableTexture<1> blended_depth,
    uniform DifferentiableTexture<1> rendered_quality_per_input,
    uniform DifferentiableTexture<1> blended_quality,
    uniform DifferentiableTexture<4> loss,
    uniform float scale_loss,
) {
    int width, height;
    rendered_depth_per_input.GetDimensions(width, height);
    if ((threadIdx.x < 0) || (threadIdx.x >= width) || (threadIdx.y < 0) || (threadIdx.y >= height)) { return;}

    float4 l = depth_consistency_loss(rendered_depth_per_input, blended_depth, rendered_quality_per_input, blended_quality, threadIdx.xy, width, height);

    loss.StoreTexel(
        threadIdx.xy,
        scale_loss * l
    );

    bwd_diff(depth_consistency_loss)(rendered_depth_per_input, blended_depth, rendered_quality_per_input, blended_quality, threadIdx.xy, width, height, scale_loss);
};