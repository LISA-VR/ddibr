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

/**
* Store all `struct` used for passing the Uniforms
*
*/

import Projection;
import DiffTex;
import DiffBuffer;
import _Constants;

struct CameraParameters {
    int proj_type;
    int2 imgSize;
    float2 min_max_depth;

    CsteBuffer param;
    float2 gFocal() { return param.load<2>(0);};
    float2 gPrincPoint() { return param.load<2>(2);};
    float3 gPos() { return param.load<3>(4);};
    float3 gRot() { return param.load<3>(7);};
};

struct DiffCameraParameters {
    int proj_type;
    int2 imgSize;
    float2 min_max_depth;

    ChunkedDiffBuffer param;

    [BackwardDifferentiable]
    float2 gFocal(int2 idxBlock) { return param.load<2>(0, idxBlock);};
    void sFocal(int2 idxBlock, float2 val) { param.save<2>(0, idxBlock, val);};

    [BackwardDifferentiable]
    float2 gPrincPoint(int2 idxBlock) { return param.load<2>(2, idxBlock);};
    void sPrincPoint(int2 idxBlock, float2 val) { param.save<2>(2, idxBlock, val);};

    [BackwardDifferentiable]
    float3 gPos(int2 idxBlock) { return param.load<3>(4, idxBlock);};
    void sPos(int2 idxBlock, float3 val) { param.save<3>(4, idxBlock, val);};

    [BackwardDifferentiable]
    float3 gRot(int2 idxBlock) { return param.load<3>(7, idxBlock);};
    void sRot(int2 idxBlock, float3 val) { param.save<3>(7, idxBlock, val);};
};

struct OptimizationParameters {
    float depth_check_tol;
}

struct QualityParameter {
    float lambda_quality_shape;
    float lambda_quality_normal;
    float lambda_quality_depth;
    float lambda_quality_side;
    float lambda_quality_foreground;
    float lambda_quality_face_camera;

    float scaling_power_quality_shape;
    float scaling_power_quality_normal;
    float scaling_power_quality_depth;
    float scaling_power_quality_side;
    float scaling_power_quality_foreground;
    float scaling_power_quality_face_camera;

    float quality_treshold;
}

struct CopyDrawResult {
    Texture2DMS<float4> ms_color;   // c_i
    Texture2DMS<float1> ms_depth;   // d_i
    Texture2DMS<float1> ms_quality; // q_i
    Texture2DMS<float1> ms_mask;    // m_i

    DifferentiableTexture<4> color;   // $ C = \sum(c_s) / S$
    DifferentiableTexture<1> depth;   // $D=\sum(d_s) / S$
    DifferentiableTexture<1> quality; // $Q=\sum(q_s) / S$
    DifferentiableTexture<1> mask;    // $M=\sum(m_s) / S$
}

struct BlendingResult {

    DifferentiableTexture<4> color;   // $ C = \sum(q_i c_i) / \sum(q_i)$
    DifferentiableTexture<1> depth;   // $D=(\sum q_i d_i) / (\sum q_i)$
    DifferentiableTexture<1> quality; // $Q=\sum q_i$
}

struct Camera {
    ConstantBuffer<CameraParameters> params;
    CstTexture<1> depth_texture;
    CstTexture<4> color_texture;
    SamplerState sampler_tex;
};

struct DiffCamera {
    ConstantBuffer<DiffCameraParameters> params;
    ConstantBuffer<DifferentiableTexture<1>> depth_texture;
    ConstantBuffer<DifferentiableTexture<4>> color_texture;
    SamplerState sampler_tex;
};

struct SpaceTransform {

    ChunkedDiffBuffer R_t;
    [BackwardDifferentiable] float3x3 g_R(int2 idxBlock) { return R_t.load<3, 3>(0, idxBlock);};
    [BackwardDifferentiable] float3 g_t(int2 idxBlock) { return R_t.load<3>(9, idxBlock);};
};

struct GradientDescentOptimizerUniforms {
    float learning_rate;
    Buffer<float> bufferLR;
}

struct GradientDescentTex4Uniforms {
    DifferentiableTexture<4> diff_tex;
};

struct GradientDescentTex1Uniforms {
    DifferentiableTexture<1> diff_tex;
};

struct AdamOptimizerUniforms {
    float EPS;
    float beta_1;
    float beta_2;
    float lr;
    float iteration;
    RWBuffer<float> bufferM1;
    RWBuffer<float> bufferM2;
    Buffer<float>   bufferLR;

    static float UpdateM1(in float m1, in float grad, in float beta_1) {
        return (beta_1 * m1) + (1.f - beta_1) * grad;
    };

    static float UpdateM2(in float m2, in float grad, in float beta_2) {
        return (beta_2 * m2) + (1.f - beta_2) * grad * grad;
    };

    static float CorrectBias(in float m, in float beta, in float it) {
        return m / (1.F - pow(beta, it));
    };

    static float GetCorrection(in float lr, in float unBiasedM1, in float unBiasedM2) {
        constexpr float EPS = cst_type_value<float>.EPS();
        return -(lr * unBiasedM1) / (sqrt(unBiasedM2) + EPS);
    }
};

struct AdamTex4Uniforms {
    DifferentiableTexture<4> diff_tex;
};

struct AdamTex1Uniforms {
    DifferentiableTexture<1> diff_tex;
};

struct AdamSmallBufferUniforms {
    uniform ConstantBuffer<DiffBuffer> diff_buffer;
    uniform StructuredBuffer<int32_t> selector;
};