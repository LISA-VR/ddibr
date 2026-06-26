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
import Uniforms;


[shader("compute")]
[NumThreads(32, 32, 1)]
void gradient_descent_tex4_main(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform ConstantBuffer<GradientDescentTex4Uniforms> learn_data,
    uniform ConstantBuffer<GradientDescentOptimizerUniforms> opti_data,
) {
    uint x = threadIdx.x;
    uint y = threadIdx.y;
    uint dstWidth, dstHeight;

    learn_data.diff_tex.GetDimensions(dstWidth, dstHeight);
    if (x >= dstWidth) return;
    if (y >= dstHeight) return;

    var val = learn_data.diff_tex.LoadGrad(uint2(x, y), dstWidth);
    learn_data.diff_tex.AddTexel(uint2(x, y), dstWidth, dstHeight, -val * opti_data.learning_rate);
}

[shader("compute")]
[NumThreads(32, 32, 1)]
void gradient_descent_tex1_main(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform ConstantBuffer<GradientDescentTex1Uniforms> learn_data,
    uniform ConstantBuffer<GradientDescentOptimizerUniforms> opti_data
) {
    uint x = threadIdx.x;
    uint y = threadIdx.y;
    uint dstWidth, dstHeight;

    learn_data.diff_tex.GetDimensions(dstWidth, dstHeight);
    if (x >= dstWidth) return;
    if (y >= dstHeight) return;

    var val = learn_data.diff_tex.LoadGrad(uint2(x, y), dstWidth);
    learn_data.diff_tex.AddTexel(uint2(x, y), dstWidth, dstHeight, -val * opti_data.learning_rate);
}

[shader("compute")]
[NumThreads(32, 32, 1)]
void adam_tex4_main(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform ConstantBuffer<AdamTex4Uniforms> learn_data,
    uniform ConstantBuffer<AdamOptimizerUniforms> opti_data
) {
    uint x = threadIdx.x;
    uint y = threadIdx.y;
    uint dstWidth, dstHeight;

    learn_data.diff_tex.GetDimensions(dstWidth, dstHeight);
    if (x >= dstWidth) return;
    if (y >= dstHeight) return;

    float4 res = float4(0.f);
    float4 grad = learn_data.diff_tex.LoadGrad(uint2(x, y), dstWidth);

    int nbChannels = 4;
    float4 m1 = float4(0.0f);
    float4 m2 = float4(0.0f);

    [ForceUnroll]
    for (uint i = 0; i < nbChannels; i++) {
        uint locMomentum = (x + dstWidth * y) * nbChannels + i;
        
        m1[i] = opti_data.bufferM1.Load(locMomentum);
        //m2[i] = opti_data.bufferM2.Load(locMomentum);
    }

    [ForceUnroll]
    for (uint i = 0; i < nbChannels; i++) {
        uint locMomentum = (x + dstWidth * y) * nbChannels + i;

        //m1[i] = opti_data.bufferM1.Load(locMomentum);
        m2[i] = opti_data.bufferM2.Load(locMomentum);
    }

    [ForceUnroll]
    for (uint i = 0; i < nbChannels; i++) {
        float g_ = grad[i];

        m1[i] = opti_data.UpdateM1(m1[i], g_, opti_data.beta_1);
        m2[i] = opti_data.UpdateM2(m2[i], g_, opti_data.beta_2);

        float m1_ = opti_data.CorrectBias(m1[i], opti_data.beta_1, opti_data.iteration);
        float m2_ = opti_data.CorrectBias(m2[i], opti_data.beta_2, opti_data.iteration);

        float r_ = opti_data.GetCorrection(opti_data.lr, m1_, m2_);

        res[i] = r_;
    }

    [ForceUnroll]
    for (uint i = 0; i < nbChannels; i++) {
        uint locMomentum = (x + dstWidth * y) * nbChannels + i;

        opti_data.bufferM1.Store(locMomentum, m1[i]);
        //opti_data.bufferM2.Store(locMomentum, m2[i]);
    }

    [ForceUnroll]
    for (uint i = 0; i < nbChannels; i++) {
        uint locMomentum = (x + dstWidth * y) * nbChannels + i;

        //opti_data.bufferM1.Store(locMomentum, m1[i]);
        opti_data.bufferM2.Store(locMomentum, m2[i]);
    }

    learn_data.diff_tex.AddTexel(uint2(x, y), dstWidth, dstHeight, res);
}

[shader("compute")]
[NumThreads(32, 32, 1)]
void adam_tex1_main(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform ConstantBuffer<AdamTex1Uniforms> learn_data,
    uniform ConstantBuffer<AdamOptimizerUniforms> opti_data
) {
    uint x = threadIdx.x;
    uint y = threadIdx.y;
    uint dstWidth, dstHeight;

    learn_data.diff_tex.GetDimensions(dstWidth, dstHeight);
    if (x >= dstWidth) return;
    if (y >= dstHeight) return;

    float res = 0.f;
    float grad = learn_data.diff_tex.LoadGrad(uint2(x, y), dstWidth).x;

    int nbChannels = 1;
    float m1 = 0.0f;
    float m2 = 0.0f;

    uint locMomentum = (x + dstWidth * y) * nbChannels;

    m1 = opti_data.bufferM1.Load(locMomentum);
    m2 = opti_data.bufferM2.Load(locMomentum);

    float g_ = grad;

    m1 = opti_data.UpdateM1(m1, g_, opti_data.beta_1);
    m2 = opti_data.UpdateM2(m2, g_, opti_data.beta_2);

    float m1_ = opti_data.CorrectBias(m1, opti_data.beta_1, opti_data.iteration);
    float m2_ = opti_data.CorrectBias(m2, opti_data.beta_2, opti_data.iteration);

    float r_ = opti_data.GetCorrection(opti_data.lr, m1_, m2_);

    res = r_;

    opti_data.bufferM1.Store(locMomentum, m1);
    opti_data.bufferM2.Store(locMomentum, m2);

    learn_data.diff_tex.AddTexel(uint2(x, y), dstWidth, dstHeight, res);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void adam_small_buffer_main(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform ConstantBuffer<AdamSmallBufferUniforms> learn_data,
    uniform ConstantBuffer<AdamOptimizerUniforms> opti_data
) {
    uint x = threadIdx.x;
    if (x > learn_data.diff_buffer.buff_.size) { return;}

    // check if parameter was selected
    int is_selected = learn_data.selector.Load(x);
    if (is_selected != 1) { return;}

    float res = 0.f;
    float grad = learn_data.diff_buffer.load_grad<1>(x).x;
    float lr = opti_data.bufferLR.Load(x);

    float m1 = 0.0f;
    float m2 = 0.0f;

    m1 = opti_data.bufferM1.Load(x);
    m2 = opti_data.bufferM2.Load(x);

    float g_ = grad;

    m1 = opti_data.UpdateM1(m1, g_, opti_data.beta_1);
    m2 = opti_data.UpdateM2(m2, g_, opti_data.beta_2);

    float m1_ = opti_data.CorrectBias(m1, opti_data.beta_1, opti_data.iteration);
    float m2_ = opti_data.CorrectBias(m2, opti_data.beta_2, opti_data.iteration);

    float r_ = opti_data.GetCorrection(lr, m1_, m2_);

    res = r_;

    opti_data.bufferM1.Store(x, m1);
    opti_data.bufferM2.Store(x, m2);

    learn_data.diff_buffer.save<1>(x, learn_data.diff_buffer.load<1>(x).x + res);
}