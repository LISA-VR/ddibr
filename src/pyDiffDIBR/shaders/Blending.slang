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

import _Constants;
import DiffTex;

void blend_view_t4_t1(
    DifferentiableTexture<4> view_to_add_t4,
    DifferentiableTexture<1> view_to_add_t1,
    DifferentiableTexture<1> quality_map,
    DifferentiableTexture<4> blended_view_t4,
    DifferentiableTexture<1> blended_view_t1,
    DifferentiableTexture<1> blended_quality,
    int2 pixel_coords, int2 sizeImg
) {
    constexpr float EPS = cst_type_value<float>.EPS();
    float quality = quality_map.LoadTexel(pixel_coords, sizeImg.x, sizeImg.y).x;
    float depth = view_to_add_t1.LoadTexel(pixel_coords, sizeImg.x, sizeImg.y).x;
    float4 color = view_to_add_t4.LoadTexel(pixel_coords, sizeImg.x, sizeImg.y);

    float4 c = quality * color;
    float d  = quality * depth;


    blended_view_t4.AddTexel(
        pixel_coords, sizeImg.x, sizeImg.y,
        c
    );

    blended_view_t1.AddTexel(
        pixel_coords, sizeImg.x, sizeImg.y,
        d
    );

    blended_quality.AddTexel(
        pixel_coords, sizeImg.x, sizeImg.y,
        quality
    );
};

void blend_view_t4_t1_with_mask(
    DifferentiableTexture<4> view_to_add_t4,
    DifferentiableTexture<1> view_to_add_t1,
    DifferentiableTexture<1> quality_map,
    DifferentiableTexture<1> mask_map,
    DifferentiableTexture<1> normal_map,
    DifferentiableTexture<4> blended_view_t4,
    DifferentiableTexture<1> blended_view_t1,
    DifferentiableTexture<1> blended_quality,
    DifferentiableTexture<1> blended_mask,
    DifferentiableTexture<1> blended_normal,
    int2 pixel_coords, int2 sizeImg
) {
    constexpr float EPS = cst_type_value<float>.EPS();
    float quality = quality_map.LoadTexel(pixel_coords, sizeImg.x, sizeImg.y).x;
    float mask = mask_map.LoadTexel(pixel_coords, sizeImg.x, sizeImg.y).x;
    float depth = view_to_add_t1.LoadTexel(pixel_coords, sizeImg.x, sizeImg.y).x;
    float4 color = view_to_add_t4.LoadTexel(pixel_coords, sizeImg.x, sizeImg.y);
    float normal = normal_map.LoadTexel(pixel_coords, sizeImg.x, sizeImg.y).x;

    float alpha = color[3];
    quality *= (alpha);

    float4 c = float4(quality * color.xyz, 1.0f);
    float d  = quality * depth;
    float n  = quality * normal;

    blended_view_t4.AddTexel(
        pixel_coords, sizeImg.x, sizeImg.y,
        c
    );

    blended_view_t1.AddTexel(
        pixel_coords, sizeImg.x, sizeImg.y,
        d
    );

    blended_quality.AddTexel(
        pixel_coords, sizeImg.x, sizeImg.y,
        quality
    );
    
    blended_mask.AddTexel(
        pixel_coords, sizeImg.x, sizeImg.y,
        mask
    );
    
    blended_normal.AddTexel(
        pixel_coords, sizeImg.x, sizeImg.y,
        n
    );
};

[Differentiable]
float normalize_mask(float mask){
    return pow<float>(min(1.F, mask), 2.0f);
}

void normalize_blending_t4_t1(
    DifferentiableTexture<4> blended_view_t4,
    DifferentiableTexture<1> blended_view_t1,
    DifferentiableTexture<1> blended_quality,
    DifferentiableTexture<1> blended_mask,
    DifferentiableTexture<1> blended_normal,
    int2 pixel_coords, int2 sizeImg,
    float4 default_value_t4 = float4(0.0f),
    float1 default_value_t1 = float1(0.f),
    float  quality_treshold = 0.f,
) {
    constexpr float EPS = cst_type_value<float>.HIGH_EPS();
    float quality = (blended_quality.LoadTexel(pixel_coords, sizeImg.x, sizeImg.y).x);
    float mask = (blended_mask.LoadTexel(pixel_coords, sizeImg.x, sizeImg.y).x);

    float4 norm_color = default_value_t4;
    float1 norm_depth = default_value_t1;
    float1 norm_normal= 0.f;
    
    if (quality > EPS) {
        mask = normalize_mask(mask);
        float total_quality = quality * mask + quality_treshold;
        float4 color = blended_view_t4.LoadTexel(pixel_coords, sizeImg.x, sizeImg.y);
        norm_color = (quality_treshold * norm_color + color) / (total_quality);

        float1 depth = blended_view_t1.LoadTexel(pixel_coords, sizeImg.x, sizeImg.y);
        norm_depth = (quality_treshold * norm_depth + depth) / (total_quality);
        
        float1 normal = blended_normal.LoadTexel(pixel_coords, sizeImg.x, sizeImg.y);
        norm_normal = (normal) / (quality * mask);
    }

    blended_view_t4.StoreTexel(
        pixel_coords,
        norm_color
    );

    blended_view_t1.StoreTexel(
        pixel_coords,
        norm_depth
    );
    
    blended_normal.StoreTexel(
        pixel_coords,
        norm_normal
    );
};

[shader("compute")]
[NumThreads(32, 32, 1)]
void main_blend_t4_t1_view(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform DifferentiableTexture<4> view_to_add_t4,
    uniform DifferentiableTexture<1> view_to_add_t1,
    uniform DifferentiableTexture<1> quality_map,
    uniform DifferentiableTexture<4> blended_view_t4,
    uniform DifferentiableTexture<1> blended_view_t1,
    uniform DifferentiableTexture<1> blended_quality,
) {
    int width, height;
    blended_view_t1.GetDimensions(width, height);
    if ((threadIdx.x < 0) || (threadIdx.x >= width) || (threadIdx.y < 0) || (threadIdx.y >= height)) { return;}

    blend_view_t4_t1(view_to_add_t4, view_to_add_t1, quality_map, blended_view_t4, blended_view_t1, blended_quality, threadIdx.xy, int2(width, height));
};

[shader("compute")]
[NumThreads(32, 32, 1)]
void main_masked_blend_t4_t1_view(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform DifferentiableTexture<4> view_to_add_t4,
    uniform DifferentiableTexture<1> view_to_add_t1,
    uniform DifferentiableTexture<1> quality_map,
    uniform DifferentiableTexture<1> mask_map,
    uniform DifferentiableTexture<1> normal_map,
    uniform DifferentiableTexture<4> blended_view_t4,
    uniform DifferentiableTexture<1> blended_view_t1,
    uniform DifferentiableTexture<1> blended_quality,
    uniform DifferentiableTexture<1> blended_mask,
    uniform DifferentiableTexture<1> blended_normal,
) {
    int width, height;
    blended_view_t1.GetDimensions(width, height);
    if ((threadIdx.x < 0) || (threadIdx.x >= width) || (threadIdx.y < 0) || (threadIdx.y >= height)) { return;}

    blend_view_t4_t1_with_mask(
        view_to_add_t4, 
        view_to_add_t1, 
        quality_map, 
        mask_map,
        normal_map, 
        blended_view_t4, 
        blended_view_t1, 
        blended_quality, 
        blended_mask,
        blended_normal,
        threadIdx.xy, 
        int2(width, height)
    );
};

[shader("compute")]
[NumThreads(32, 32, 1)]
void main_normalize_t4_t1_blending(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform DifferentiableTexture<4> blended_view_t4,
    uniform DifferentiableTexture<1> blended_view_t1,
    uniform DifferentiableTexture<1> blended_quality,
    uniform DifferentiableTexture<1> blended_mask,
    uniform DifferentiableTexture<1> blended_normal,
    uniform float4 default_value_t4,
    uniform float1 default_value_t1,
    uniform float1 quality_treshold,
) {
    int width, height;
    blended_view_t1.GetDimensions(width, height);
    if ((threadIdx.x < 0) || (threadIdx.x >= width) || (threadIdx.y < 0) || (threadIdx.y >= height)) { return;}

    normalize_blending_t4_t1(blended_view_t4, blended_view_t1, blended_quality, blended_mask, blended_normal, threadIdx.xy, int2(width, height), default_value_t4, default_value_t1, quality_treshold);
};


[shader("compute")]
[NumThreads(32, 32, 1)]
void main_bwd_blending_t4_t1(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform DifferentiableTexture<4> color,
    uniform DifferentiableTexture<1> depth,
    uniform DifferentiableTexture<1> quality_map,
    uniform DifferentiableTexture<1> blended_quality,
    uniform DifferentiableTexture<4> blended_color,
    uniform DifferentiableTexture<1> blended_depth,
) {
    int width, height;
    blended_color.GetDimensions(width, height);
    if ((threadIdx.x < 0) || (threadIdx.x >= width) || (threadIdx.y < 0) || (threadIdx.y >= height)) { return;}
    
    // if quality is 0, then we discard gradient 
    float quality = blended_quality.LoadTexel(threadIdx.xy, width, height).x;
    constexpr float EPS = cst_type_value<float>.EPS();
    if (quality < EPS) { return;}

    // 1. Diff $\partial C / \partial c_i$
    bwd_dX_dxi<4>(
        color,
        quality_map,
        blended_quality,
        blended_color,
        threadIdx.xy, width, height
    );

    // 2. Diff $\partial D / \partial d_i$
    bwd_dX_dxi<1>(
        depth,
        quality_map,
        blended_quality,
        blended_depth,
        threadIdx.xy, width, height
    );

    // 3. Diff $\partial C / \partial q_i$
    bwd_dX_dqi<4>(
        color,
        quality_map,
        blended_quality,
        blended_color,
        threadIdx.xy, width, height
    );

    // 4. Diff $\partial D / \partial q_i$
    bwd_dX_dqi<1>(
        depth,
        quality_map,
        blended_quality,
        blended_depth,
        threadIdx.xy, width, height
    );
};

[shader("compute")]
[NumThreads(32, 32, 1)]
void main_bwd_masked_blending_t4_t1(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform DifferentiableTexture<4> color,
    uniform DifferentiableTexture<1> depth,
    uniform DifferentiableTexture<1> quality_map,
    uniform DifferentiableTexture<1> mask_map,
    uniform DifferentiableTexture<4> blended_color,
    uniform DifferentiableTexture<1> blended_depth,
    uniform DifferentiableTexture<1> blended_quality,
    uniform DifferentiableTexture<1> blended_mask,
    uniform float4 default_value_t4,
    uniform float1 default_value_t1,
    uniform float1 quality_treshold,
) {
    int width, height;
    blended_color.GetDimensions(width, height);
    if ((threadIdx.x < 0) || (threadIdx.x >= width) || (threadIdx.y < 0) || (threadIdx.y >= height)) { return;}

    // if quality is 0, then we discard gradient
    float quality = blended_quality.LoadTexel(threadIdx.xy, width, height).x;
    float m       = blended_mask.LoadTexel(threadIdx.xy, width, height).x;
    float q_i     = quality_map.LoadTexel(threadIdx.xy, width, height).x;
    float m_i     = mask_map.LoadTexel(threadIdx.xy, width, height).x;
    
    constexpr float EPS = cst_type_value<float>.HIGH_EPS();
    if ((quality < EPS) || (m < EPS) || (q_i < EPS) || (m_i < EPS)) { return;} // Occlusions or empty regions: do not propagate any gradients.

    float4 c_i = color.LoadTexel(threadIdx.xy, width, height);
    float a_i = c_i[3];
    float d_i = depth.LoadTexel(threadIdx.xy, width, height).x;

    float4 c = blended_color.LoadTexel(threadIdx.xy, width, height);
    float  d = blended_depth.LoadTexel(threadIdx.xy, width, height).x;
    //float q = blended_quality.LoadTexel(threadIdx.xy, width, height).x;
    float m_prime = normalize_mask(m);
    
    float4 grad_c = blended_color.LoadGrad(threadIdx.xy, width);
    float  grad_d = blended_depth.LoadGrad(threadIdx.xy, width).x;
    float  grad_q = blended_quality.LoadGrad(threadIdx.xy, width).x;
    

    // 1. Diff $\partial C / \partial c_i$
    float3 grad_ci = grad_dX_dxi<3>(
        c_i.xyz,
        a_i,
        q_i,
        quality_treshold,
        m_prime,
        quality,
        grad_c.xyz,
        threadIdx.xy, width, height
    );
    float1 grad_dc_dai = grad_dX_dai<3>(
        c_i.xyz,
        q_i,
        m_prime,
        quality,
        quality_treshold,
        c.xyz,
        grad_c.xyz,
        threadIdx.xy, width, height
    );
    // 4. Diff $\partial D / \partial q_i$
    float1 grad_dd_dai = grad_dX_dai<1>(
        d_i,
        q_i,
        m_prime,
        quality,
        quality_treshold,
        d,
        grad_d,
        threadIdx.xy, width, height
    );
    float grad_dq_dai = grad_dQ_dai(
        q_i,
        grad_q,
        threadIdx.xy, width, height
    );  
    
    color.bwd_LoadTexel(threadIdx.xy, width, height, float4(grad_ci, grad_dc_dai.x + grad_dd_dai.x + grad_dq_dai));

    // 2. Diff $\partial D / \partial d_i$
    float1 grad_di = grad_dX_dxi<1>(
        d_i,
        a_i,
        q_i,
        quality_treshold,
        m_prime,
        quality,
        grad_d,
        threadIdx.xy, width, height
    );
    depth.bwd_LoadTexel(threadIdx.xy, width, height, grad_di);

    // 3. Diff $\partial C / \partial q_i$
    float1 grad_dc_dqi = grad_dX_dqi<3>(
        c_i.xyz,
        a_i,
        m_prime,
        quality,
        quality_treshold,
        c.xyz,
        grad_c.xyz,
        threadIdx.xy, width, height
    );
    // 4. Diff $\partial D / \partial q_i$
    float1 grad_dd_dqi = grad_dX_dqi<1>(
        d_i,
        a_i,
        m_prime,
        quality,
        quality_treshold,
        d,
        grad_d,
        threadIdx.xy, width, height
    );
    float grad_dq_dqi = grad_dQ_dqi(
        a_i,
        grad_q,
        threadIdx.xy, width, height
    );  
    quality_map.bwd_LoadTexel(threadIdx.xy, width, height, grad_dc_dqi + grad_dd_dqi + grad_dq_dqi);

    // 3. Diff $\partial C / \partial q_i$
    float1 grad_dc_dmi = grad_dX_dmi<3>(
        m,
        m_prime,
        quality,
        quality_treshold,
        c.xyz,
        grad_c.xyz,
        threadIdx.xy, width, height
    );
    // 4. Diff $\partial D / \partial q_i$
    float1 grad_dd_dmi = grad_dX_dmi<1>(
        m,
        m_prime,
        quality,
        quality_treshold,
        d,
        grad_d,
        threadIdx.xy, width, height
    );
    mask_map.bwd_LoadTexel(threadIdx.xy, width, height, grad_dc_dmi + grad_dd_dmi);
    
    
};

/**
$$
C = \sum(q_i c_i) / (\sum q_i)
$$ $$
\partial C / \partial c_j = (q_j) / (\sum q_i) = (q_j) / Q
$$
*/
void bwd_dC_dci(
    DifferentiableTexture<4> color,
    DifferentiableTexture<1> quality_map,
    DifferentiableTexture<1> blended_quality,
    DifferentiableTexture<4> blended_color,
    int2 loc,
    int w,
    int h
) {
    //float4 c_j  = color.LoadTexel(loc, w, h);
    float q_j = quality_map.LoadTexel(loc, w, h).x;
    float Q = blended_quality.LoadTexel(loc, w, h).x;
    constexpr float EPS = cst_type_value<float>.EPS();

    float4 grad = (q_j) / (Q + EPS);
    float4 input_grad = blended_color.LoadGrad(loc, w);
    float4 grad_ = input_grad * grad;

    color.bwd_LoadTexel(loc, w, h, grad_);
};

/**
$$
D = \sum(q_i d_i) / (\sum q_i)
$$ $$
\partial D / \partial d_j = (q_j) / (\sum q_i) = (q_j) / Q
$$
*/
void bwd_dD_ddi(
    DifferentiableTexture<1> depth,
    DifferentiableTexture<1> quality_map,
    DifferentiableTexture<1> blended_quality,
    DifferentiableTexture<1> blended_depth,
    int2 loc,
    int w,
    int h
) {
    //float d_j = depth.LoadTexel(loc, w, h).x;
    float q_j = quality_map.LoadTexel(loc, w, h).x;
    float Q = blended_quality.LoadTexel(loc, w, h).x;
    constexpr float EPS = cst_type_value<float>.EPS();

    float grad = (q_j) / (Q + EPS);
    float input_grad = blended_depth.LoadGrad(loc, w).x;

    depth.bwd_LoadTexel(loc, w, h, input_grad * grad);
};

void bwd_dX_dxi<let N : int>(
    DifferentiableTexture<N> x,
    DifferentiableTexture<1> quality_map,
    DifferentiableTexture<1> blended_quality,
    DifferentiableTexture<N> blended_x,
    int2 loc,
    int w,
    int h
) {
    typedef vector<float, N> VectorType_;

    float q_j = quality_map.LoadTexel(loc, w, h).x;
    float Q = blended_quality.LoadTexel(loc, w, h).x;
    constexpr float EPS = cst_type_value<float>.EPS();

    VectorType_ grad = (q_j) / (Q + EPS);
    VectorType_ input_grad = blended_x.LoadGrad(loc, w);
    VectorType_ grad_ = input_grad * grad;

    x.bwd_LoadTexel(loc, w, h, grad_);
};

vector<float, N> grad_dX_dxi<let N : int>(
    vector<float, N> x_i,
    float            a_i,
    float            q_i,
    float            q_0,
    vector<float, 1> m_prime,
    vector<float, 1> q,
    vector<float, N> grad_x,
    int2 loc,
    int w,
    int h
) {
    typedef vector<float, N> VectorType_;
    
    VectorType_ grad_xi = (a_i * q_i) / (m_prime * a_i * q_i + q_0); 
    grad_xi *= grad_x;

    return grad_xi;
};

vector<float, 1> grad_dX_dai<let N : int>(
    vector<float, N>    x_i,
    float               q_i,
    float               m_prime,
    float1              q,
    float1              q_0,
    vector<float, N>    x,
    vector<float, N>    grad_x,
    int2 loc,
    int w,
    int h
) {
    typedef vector<float, N> VectorType_;
    
    VectorType_ numerator = x;
    VectorType_ denominator = m_prime * q + q_0;
    
    VectorType_ grad = (q_i * x_i) / denominator - numerator * m_prime * q_i / denominator;
    return mul(grad_x, grad);
};

float grad_dQ_dai(
    float    q_i,
    float    grad_q,
    int2 loc,
    int w,
    int h
) {
    return grad_q * q_i;
};

float grad_dQ_dqi(
    float    a_i,
    float    grad_q,
    int2 loc,
    int w,
    int h
) {
    return grad_q * a_i;
};


/**
$$
C = \sum(q_i c_i) / (\sum q_i)
$$ $$
\partial C / \partial q_j = dq_j \left(c_j / Q - (C \: Q) / q^2_j \right)
$$
*/
void bwd_dC_dqi(
    DifferentiableTexture<4> color,
    DifferentiableTexture<1> quality_map,
    DifferentiableTexture<1> blended_quality,
    DifferentiableTexture<4> blended_color,
    int2 loc,
    int w,
    int h
) {
    float4 c_j  = color.LoadTexel(loc, w, h);
    float q_j   = quality_map.LoadTexel(loc, w, h).x;
    float Q     = blended_quality.LoadTexel(loc, w, h).x;
    float4 C = blended_color.LoadTexel(loc, w, h);
    constexpr float EPS = cst_type_value<float>.EPS();
    if (Q < EPS) { return;}

    // float4 grad = c_j / (Q + EPS) - ((C * Q) / (q_j * q_j + EPS));
    float4 grad = (c_j - C) / (Q + EPS);
    float4 input_grad = blended_color.LoadGrad(loc, w);
    float grad_ = mul(input_grad, grad);

    quality_map.bwd_LoadTexel(loc, w, h, grad_);
};

void bwd_dX_dqi<let N : int>(
    DifferentiableTexture<N> x,
    DifferentiableTexture<1> quality_map,
    DifferentiableTexture<1> blended_quality,
    DifferentiableTexture<N> blended_x,
    int2 loc,
    int w,
    int h
) {
    typedef vector<float, N> VectorType_;

    VectorType_ x_j = x.LoadTexel(loc, w, h);
    float q_j = quality_map.LoadTexel(loc, w, h).x;
    float Q = blended_quality.LoadTexel(loc, w, h).x;
    VectorType_ X = blended_x.LoadTexel(loc, w, h);
    constexpr float EPS = cst_type_value<float>.EPS();
    if (Q < EPS) { return;}

    VectorType_ grad = (x_j - X) / (Q + EPS);
    VectorType_ input_grad = blended_x.LoadGrad(loc, w);
    float grad_ = mul(input_grad, grad);

    quality_map.bwd_LoadTexel(loc, w, h, grad_);
};

vector<float, 1> grad_dX_dqi<let N : int>(
    vector<float, N>    x_i,
    float               a_i,
    float               m_prime,
    float1              q,
    float1              q_0,
    vector<float, N>    x,
    vector<float, N>    grad_x,
    int2 loc,
    int w,
    int h
) {
    typedef vector<float, N> VectorType_;
    
    VectorType_ numerator = x;
    VectorType_ denominator = m_prime * q + q_0;
    
    VectorType_ grad = (a_i * x_i) / denominator - numerator * m_prime * a_i / denominator;
    return mul(grad_x, grad);
};

vector<float, 1> grad_dX_dmi<let N : int>(
    float               m,
    float               m_prime,
    float               q,
    float               q_0,
    vector<float, N>    x,
    vector<float, N>    grad_x,
    int2 loc,
    int w,
    int h
) {
    typedef vector<float, N> VectorType_;
    VectorType_ denominator = m_prime * q + q_0;
    
    VectorType_ grad = - x / (denominator) * q;
    float grad_ = mul(grad_x, grad);
    
    DifferentialPair<float> d_m = diffPair(m);
    bwd_diff(normalize_mask)(d_m, grad_);
    return d_m.d;
};

/**
$$
C = \sum(q_i c_i) / (\sum q_i)
$$ $$
\partial C / \partial q_j = dq_j \left(c_j / Q - (C \: Q) / q^2_j \right)
$$
*/
void bwd_dX_dqi_with_mask<let N : int>(
    DifferentiableTexture<N> x,
    DifferentiableTexture<1> quality_map,
    DifferentiableTexture<1> mask_map,
    DifferentiableTexture<1> blended_quality,
    DifferentiableTexture<N> blended_x,
    int2 loc,
    int w,
    int h
) {
    typedef vector<float, N> VectorType_;

    VectorType_ x_j = x.LoadTexel(loc, w, h);
    float q_j = quality_map.LoadTexel(loc, w, h).x;
    float m_j = mask_map.LoadTexel(loc, w, h).x;
    float Q = blended_quality.LoadTexel(loc, w, h).x;
    VectorType_ X = blended_x.LoadTexel(loc, w, h);
    constexpr float EPS = cst_type_value<float>.EPS();
    if (Q < EPS) { return;}

    VectorType_ grad_qj = (x_j - m_j * X) / (Q + EPS);
    VectorType_ grad_mj = (0.f - q_j * X) / (Q + EPS);
    VectorType_ input_grad = blended_x.LoadGrad(loc, w);

    float grad_qj_ = mul(input_grad, grad_qj);
    float grad_mj_ = mul(input_grad, grad_mj);

    quality_map.bwd_LoadTexel(loc, w, h, grad_qj_);
    mask_map.bwd_LoadTexel(loc, w, h, grad_mj_);
};

/**
$$
D = \sum(q_i d_i) / (\sum q_i)
$$ $$
\partial D / \partial q_j = dq_j \left(d_j / Q - (D \: Q) / q^2_j \right)
$$
*/
void bwd_dD_dqi(
    DifferentiableTexture<1> depth,
    DifferentiableTexture<1> quality_map,
    DifferentiableTexture<1> blended_quality,
    DifferentiableTexture<1> blended_depth,
    int2 loc,
    int w,
    int h
) {
    float d_j = depth.LoadTexel(loc, w, h).x;
    float q_j = quality_map.LoadTexel(loc, w, h).x;
    float Q = blended_quality.LoadTexel(loc, w, h).x;
    float D = blended_depth.LoadTexel(loc, w, h).x;
    constexpr float EPS = cst_type_value<float>.EPS();
    if (Q < EPS) { return;}

    float grad = (d_j - D) / (Q + EPS);
    float input_grad = blended_depth.LoadGrad(loc, w);
    float grad_ = input_grad * grad;

    quality_map.bwd_LoadTexel(loc, w, h, grad_);
};
