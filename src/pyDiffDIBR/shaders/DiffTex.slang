/*
 This file is based on/modified from "slang"
 which is Copyright (c) 2026 The Khronos® Group Inc. All rights reserved.
 
 Original code is licensed under the Apache 2.0 with LLVM Exception.
 Modifications and Slang port are licensed under the GNU Affero General Public License v3.0.
 
 For the Apache 2.0 with LLVM Exception terms of the original work, see https://github.com/shader-slang/slang/blob/master/LICENSE.
 For AGPL-3.0-or-later terms see below.

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

// -------------------------------------------------------------------------
// This code is inspired from a file of the 'slang' repository. 
// Link: https://github.com/shader-slang/slang/blob/vulkan-sdk-1.4.350.1/examples/autodiff-texture/reconstruct.slang
// The repository is licensed under 'Apache 2.0 WITH LLVM-exception'
// Link: https://github.com/shader-slang/slang/blob/vulkan-sdk-1.4.350.1/LICENSE
// -------------------------------------------------------------------------

import _Constants;

struct DifferentiableTexture<let N : int>
{
    typedef vector<float, N> T;

    bool no_grad;
    RWStructuredBuffer<Atomic<int32_t>> accumulateBuffer; // Gradient accumulate buffer
    RWTexture2D<T> texture;                       // Hardware texture handle.
    float range;

    [BackwardDerivative(bwd_LoadTexel)]
    T LoadTexel(uint2 location, uint w, uint h)
    {
        location.x = clamp(location.x, 0, w-1);
        location.y = clamp(location.y, 0, h-1);
        return texture.Load(location);
    }

    float gradScaling() {
        constexpr float max_int = cst_type_value<float>.MAX_SINT32_VALUE();
        constexpr float max_int_rescale = max_int / 1e4f;
        return  max_int_rescale / (range);
    };

    void bwd_LoadTexel(uint2 location, uint w, uint h, T val)
    {
        if (no_grad) { return;}

        const float MAX_VALUE = gradScaling();
        [ForceUnroll] //* Force the compiler to unroll the for loop so the SPIR-V code is not ugly as fck
        for (uint i = 0; i < N; i++) {
            accumulateBuffer[(location.y * w + location.x) * N + i] += int32_t(val[i] * MAX_VALUE);
        }
    }

    void StoreTexel(uint2 location, T val) {
        texture.Store(location, val);
    };

    void AddTexel(uint2 location, uint width, uint height, T val) {
        if(mul(T(1.f), abs(val)) <= cst_type_value<float>.EPS()){return;} 
        
        T out_val = LoadTexel(location, width, height) + val;
        StoreTexel(location, out_val);
    }

    [BackwardDerivativeOf(StoreTexel)]
    void bwd_StoreTexel(in uint2 location, inout DifferentialPair<T> val) {
        int w, h;
        texture.GetDimensions(w, h);

        val = diffPair(val.p, LoadGrad(location, w));
    }

    T LoadGrad(uint2 location, uint dLayerW) {
        const float MAX_VALUE = gradScaling();
        T val = T(0.0);
        [ForceUnroll] //* Force the compiler to unroll the for loop so the SPIR-V code is not ugly as fck
        for (uint i = 0; i < N; i++) {
            val[i] = float(accumulateBuffer[(location.y * dLayerW + location.x) * N + i].load()) / MAX_VALUE;
        }
        return val;
    };

    // Software reference implementation of linear filtering.
    [BackwardDifferentiable]
    T sampleTexture_linear(float2 uv, uint w, uint h)
    {
        uv = clamp(uv, 0.f, 1.0f);

        uv = uv - no_diff(floor(uv)); // Make sure UV is between 0-1
        float2 loc = uv * float2(w, h) - float2(0.5);
        float x0 = no_diff(floor(loc.x));
        float y0 = no_diff(floor(loc.y));
        float fracX = loc.x - x0;
        float fracY = loc.y - y0;
        float x1 = x0 + 1.f;
        float y1 = y0 + 1.f;

        // Can cause some problems so better to exclude or map to 0 - WH
        x0 = clamp(x0, 0.f, float(w)-1.f);
        x1 = clamp(x1, 0.f, float(w)-1.f);
        y0 = clamp(y0, 0.f, float(h)-1.f);
        y1 = clamp(y1, 0.f, float(h)-1.f);

        float weight0 = 1.0f - fracY;
        float weight1 = fracY;
        float weight00 = weight0 * (1.0f - fracX);
        float weight01 = weight0 * fracX;
        float weight10 = weight1 * (1.0f - fracX);
        float weight11 = weight1 * fracX;

        return 
        LoadTexel(uint2(int(x0), int(y0)), w, h) * weight00 +
        LoadTexel(uint2(int(x1), int(y0)), w, h) * weight01 +
        LoadTexel(uint2(int(x0), int(y1)), w, h) * weight10 +
        LoadTexel(uint2(int(x1), int(y1)), w, h) * weight11;
    }

    // Note that there is no need to mark this [BackwardDifferentiable] since it has a substitute
    // that is marked [BackwardDifferentiable]. The compiler automatically considers a call to
    // sample() to be differentiable.
    [BackwardDifferentiable]
    T sample(SamplerState s, float2 uv)
    {
        uint w;
        uint h;
        texture.GetDimensions(w, h);
        return sampleTexture_linear(uv, w, h);
    }

    void GetDimensions(out uint w, out uint h) {
        texture.GetDimensions(w, h);
    };
};

// todo must put function ddx, ddy, get_dady there...
struct DerivativeMode {
    static const uint None = 0;
    static const uint Sobel = 1;
    static const uint Optimal_3 = 2;
    static const uint Optimal_5 = 3;
}

struct CstTexture<let N : int>
{
    typedef vector<float, N> T;
    Texture2D<T> texture;

    T LoadTexel(uint2 location, uint w, uint h)
    {
        location.x = clamp(location.x, 0, w-1);
        location.y = clamp(location.y, 0, h-1);
        
        return texture.Load(uint3(location, 0));
    }

    T sample(SamplerState s, float2 uv)
    {
        return texture.Sample(s, uv);
    }

    void GetDimensions(out uint w, out uint h) {
        texture.GetDimensions(w, h);
    };

    T Get_ddy(uint2 uv, const uint mode = DerivativeMode.Sobel) {
        uint w;
        uint h;
        texture.GetDimensions(w, h);
        T res = 0;

        if (mode == DerivativeMode.Sobel) {
            T x_m1 =  1.f * LoadTexel(uv + uint2(-1, -1), w, h) - 1.f * LoadTexel(uv + uint2(-1, 1), w, h);
            T x_0 =   2.f * LoadTexel(uv + uint2( 0, -1), w, h) - 2.f * LoadTexel(uv + uint2( 0, 1), w, h);
            T x_p1 =  1.f * LoadTexel(uv + uint2( 1, -1), w, h) - 1.f * LoadTexel(uv + uint2( 1, 1), w, h);

            res += x_m1 + x_0 + x_p1;
        } else if (mode == DerivativeMode.Optimal_3) {
            T x_m1 = 0.425287f * LoadTexel(uv + uint2(-1, -1), w, h) - 0.425287f * LoadTexel(uv + uint2(-1, 1), w, h);
            T x_0  = 0.425287f * LoadTexel(uv + uint2( 0, -1), w, h) - 0.425287f * LoadTexel(uv + uint2( 0, 1), w, h);
            T x_p1 = 0.425287f * LoadTexel(uv + uint2( 1, -1), w, h) - 0.425287f * LoadTexel(uv + uint2( 1, 1), w, h);

            res += 0.229879f * x_m1 + 0.540242f * x_0 + 0.229879f * x_p1;
        } else if (mode == DerivativeMode.Optimal_5) {
            T x_m2 = 0.104550f * LoadTexel(uv + uint2(-2, -2), w, h) + 0.292315f * LoadTexel(uv + uint2(-2, -1), w, h) - 0.292315f * LoadTexel(uv + uint2(-2, 1), w, h) - 0.104550f * LoadTexel(uv + uint2(-2, 2), w, h);
            T x_m1 = 0.104550f * LoadTexel(uv + uint2(-1, -2), w, h) + 0.292315f * LoadTexel(uv + uint2(-1, -1), w, h) - 0.292315f * LoadTexel(uv + uint2(-1, 1), w, h) - 0.104550f * LoadTexel(uv + uint2(-1, 2), w, h);
            T x_0  = 0.104550f * LoadTexel(uv + uint2( 0, -2), w, h) + 0.292315f * LoadTexel(uv + uint2( 0, -1), w, h) - 0.292315f * LoadTexel(uv + uint2( 0, 1), w, h) - 0.104550f * LoadTexel(uv + uint2( 0, 2), w, h);
            T x_p1 = 0.104550f * LoadTexel(uv + uint2( 1, -2), w, h) + 0.292315f * LoadTexel(uv + uint2( 1, -1), w, h) - 0.292315f * LoadTexel(uv + uint2( 1, 1), w, h) - 0.104550f * LoadTexel(uv + uint2( 1, 2), w, h);
            T x_p2 = 0.104550f * LoadTexel(uv + uint2( 2, -2), w, h) + 0.292315f * LoadTexel(uv + uint2( 2, -1), w, h) - 0.292315f * LoadTexel(uv + uint2( 2, 1), w, h) - 0.104550f * LoadTexel(uv + uint2( 2, 2), w, h);

            res += 0.030320f * x_m2 + 0.249724f * x_m1 + 0.439911f * x_0 + 0.249724f * x_p1 + 0.030320f * x_p2;
        }

        return res;
    }

    T Get_ddx(uint2 uv, const uint mode = DerivativeMode.Sobel) {
        uint w;
        uint h;
        texture.GetDimensions(w, h);
        T res = 0;

        if (mode == DerivativeMode.Sobel) {
            T y_m1 = 1.f * LoadTexel(uv + uint2(-1, -1), w, h) - 1.f * LoadTexel(uv + uint2(1, -1), w, h);
            T y_0  = 2.f * LoadTexel(uv + uint2(-1,  0), w, h) - 2.f * LoadTexel(uv + uint2(1,  0), w, h);
            T y_p1 = 1.f * LoadTexel(uv + uint2(-1,  1), w, h) - 1.f * LoadTexel(uv + uint2(1,  1), w, h);

            res += y_m1 + y_0 + y_p1;
        } else if (mode == DerivativeMode.Optimal_3) {
            T y_m1 = 0.425287f * LoadTexel(uv + uint2(-1, -1), w, h) - 0.425287f * LoadTexel(uv + uint2(1, -1), w, h);
            T y_0  = 0.425287f * LoadTexel(uv + uint2(-1,  0), w, h) - 0.425287f * LoadTexel(uv + uint2(1,  0), w, h);
            T y_p1 = 0.425287f * LoadTexel(uv + uint2(-1,  1), w, h) - 0.425287f * LoadTexel(uv + uint2(1,  1), w, h);

            res += 0.229879f * y_m1 + 0.540242f * y_0 + 0.229879f * y_p1;
        } else if (mode == DerivativeMode.Optimal_5) {
            T y_m2 = 0.104550f * LoadTexel(uv + uint2(-2, -2), w, h) + 0.292315f * LoadTexel(uv + uint2(-1, -2), w, h) - 0.292315f * LoadTexel(uv + uint2(1, -2), w, h) - 0.104550f * LoadTexel(uv + uint2(2, -2), w, h);
            T y_m1 = 0.104550f * LoadTexel(uv + uint2(-2, -1), w, h) + 0.292315f * LoadTexel(uv + uint2(-1, -1), w, h) - 0.292315f * LoadTexel(uv + uint2(1, -1), w, h) - 0.104550f * LoadTexel(uv + uint2(2, -1), w, h);
            T y_0  = 0.104550f * LoadTexel(uv + uint2(-2,  0), w, h) + 0.292315f * LoadTexel(uv + uint2(-1,  0), w, h) - 0.292315f * LoadTexel(uv + uint2(1,  0), w, h) - 0.104550f * LoadTexel(uv + uint2(2,  0), w, h);
            T y_p1 = 0.104550f * LoadTexel(uv + uint2(-2,  1), w, h) + 0.292315f * LoadTexel(uv + uint2(-1,  1), w, h) - 0.292315f * LoadTexel(uv + uint2(1,  1), w, h) - 0.104550f * LoadTexel(uv + uint2(2,  1), w, h);
            T y_p2 = 0.104550f * LoadTexel(uv + uint2(-2,  2), w, h) + 0.292315f * LoadTexel(uv + uint2(-1,  2), w, h) - 0.292315f * LoadTexel(uv + uint2(1,  2), w, h) - 0.104550f * LoadTexel(uv + uint2(2,  2), w, h);

            res += 0.030320f * y_m2 + 0.249724f * y_m1 + 0.439911f * y_0 + 0.249724f * y_p1 + 0.030320f * y_p2;
        }

        return res;
    }

    float2 Get_DxDy(uint2 uv, const uint mode = DerivativeMode.Sobel) {
        return float2(mul(1.0, Get_ddx(uv, mode)), mul(1.0, Get_ddy(uv, mode)));
    }
};

struct MSCstTexture<let N : int>
{
    typedef vector<float, N> T;
    Texture2DMS<T> texture;

    T LoadTexel(uint3 locationAndSampleIdx, uint w, uint h, uint nbS)
    {
        locationAndSampleIdx.x = clamp(locationAndSampleIdx.x, 0, w-1);
        locationAndSampleIdx.y = clamp(locationAndSampleIdx.y, 0, h-1);
        locationAndSampleIdx.z = clamp(locationAndSampleIdx.z, 0, nbS-1);

        return texture.Load(locationAndSampleIdx);
    }

    void GetDimensions(out uint w, out uint h, out uint nbS) {
        texture.GetDimensions(w, h, nbS);
    };

    T Get_ddy(uint3 uvAndSampleIdx, const uint mode = DerivativeMode.Sobel) {
        uint w;
        uint h;
        uint nbS;
        texture.GetDimensions(w, h, nbS);
        T res = 0;

        if (mode == DerivativeMode.Sobel) {
            T x_m1 = 1.f * LoadTexel(uvAndSampleIdx + uint3(-1, -1, 0), w, h, nbS) - 1.f * LoadTexel(uvAndSampleIdx + uint3(-1, 1, 0), w, h, nbS);
            T x_0 = 2.f * LoadTexel(uvAndSampleIdx + uint3(0, -1, 0), w, h, nbS) - 2.f * LoadTexel(uvAndSampleIdx + uint3(0, 1, 0), w, h, nbS);
            T x_p1 = 1.f * LoadTexel(uvAndSampleIdx + uint3(1, -1, 0), w, h, nbS) - 1.f * LoadTexel(uvAndSampleIdx + uint3(1, 1, 0), w, h, nbS);

            res += x_m1 + x_0 + x_p1;
        } else if (mode == DerivativeMode.Optimal_3) {
            T x_m1 = 0.425287f * LoadTexel(uvAndSampleIdx + uint3(-1, -1, 0), w, h, nbS) - 0.425287f * LoadTexel(uvAndSampleIdx + uint3(-1, 1, 0), w, h, nbS);
            T x_0 = 0.425287f * LoadTexel(uvAndSampleIdx + uint3(0, -1, 0), w, h, nbS) - 0.425287f * LoadTexel(uvAndSampleIdx + uint3(0, 1, 0), w, h, nbS);
            T x_p1 = 0.425287f * LoadTexel(uvAndSampleIdx + uint3(1, -1, 0), w, h, nbS) - 0.425287f * LoadTexel(uvAndSampleIdx + uint3(1, 1, 0), w, h, nbS);

            res += 0.229879f * x_m1 + 0.540242f * x_0 + 0.229879f * x_p1;
        } else if (mode == DerivativeMode.Optimal_5) {
            T x_m2 = 0.104550f * LoadTexel(uvAndSampleIdx + uint3(-2, -2, 0), w, h, nbS) + 0.292315f * LoadTexel(uvAndSampleIdx + uint3(-2, -1, 0), w, h, nbS) - 0.292315f * LoadTexel(uvAndSampleIdx + uint3(-2, 1, 0), w, h, nbS) - 0.104550f * LoadTexel(uvAndSampleIdx + uint3(-2, 2, 0), w, h, nbS);
            T x_m1 = 0.104550f * LoadTexel(uvAndSampleIdx + uint3(-1, -2, 0), w, h, nbS) + 0.292315f * LoadTexel(uvAndSampleIdx + uint3(-1, -1, 0), w, h, nbS) - 0.292315f * LoadTexel(uvAndSampleIdx + uint3(-1, 1, 0), w, h, nbS) - 0.104550f * LoadTexel(uvAndSampleIdx + uint3(-1, 2, 0), w, h, nbS);
            T x_0 = 0.104550f * LoadTexel(uvAndSampleIdx + uint3(0, -2, 0), w, h, nbS) + 0.292315f * LoadTexel(uvAndSampleIdx + uint3(0, -1, 0), w, h, nbS) - 0.292315f * LoadTexel(uvAndSampleIdx + uint3(0, 1, 0), w, h, nbS) - 0.104550f * LoadTexel(uvAndSampleIdx + uint3(0, 2, 0), w, h, nbS);
            T x_p1 = 0.104550f * LoadTexel(uvAndSampleIdx + uint3(1, -2, 0), w, h, nbS) + 0.292315f * LoadTexel(uvAndSampleIdx + uint3(1, -1, 0), w, h, nbS) - 0.292315f * LoadTexel(uvAndSampleIdx + uint3(1, 1, 0), w, h, nbS) - 0.104550f * LoadTexel(uvAndSampleIdx + uint3(1, 2, 0), w, h, nbS);
            T x_p2 = 0.104550f * LoadTexel(uvAndSampleIdx + uint3(2, -2, 0), w, h, nbS) + 0.292315f * LoadTexel(uvAndSampleIdx + uint3(2, -1, 0), w, h, nbS) - 0.292315f * LoadTexel(uvAndSampleIdx + uint3(2, 1, 0), w, h, nbS) - 0.104550f * LoadTexel(uvAndSampleIdx + uint3(2, 2, 0), w, h, nbS);

            res += 0.030320f * x_m2 + 0.249724f * x_m1 + 0.439911f * x_0 + 0.249724f * x_p1 + 0.030320f * x_p2;
        }

        return res;
    }

    T Get_ddx(uint3 uvAndSampleIdx, const uint mode = DerivativeMode.Sobel) {
        uint w;
        uint h;
        uint nbS;
        texture.GetDimensions(w, h, nbS);
        T res = 0;

        if (mode == DerivativeMode.Sobel) {
            T y_m1 = 1.f * LoadTexel(uvAndSampleIdx + uint3(-1, -1, 0), w, h, nbS) - 1.f * LoadTexel(uvAndSampleIdx + uint3(1, -1, 0), w, h, nbS);
            T y_0 = 2.f * LoadTexel(uvAndSampleIdx + uint3(-1, 0, 0), w, h, nbS) - 2.f * LoadTexel(uvAndSampleIdx + uint3(1, 0, 0), w, h, nbS);
            T y_p1 = 1.f * LoadTexel(uvAndSampleIdx + uint3(-1, 1, 0), w, h, nbS) - 1.f * LoadTexel(uvAndSampleIdx + uint3(1, 1, 0), w, h, nbS);

            res += y_m1 + y_0 + y_p1;
        } else if (mode == DerivativeMode.Optimal_3) {
            T y_m1 = 0.425287f * LoadTexel(uvAndSampleIdx + uint3(-1, -1, 0), w, h, nbS) - 0.425287f * LoadTexel(uvAndSampleIdx + uint3(1, -1, 0), w, h, nbS);
            T y_0 = 0.425287f * LoadTexel(uvAndSampleIdx + uint3(-1, 0, 0), w, h, nbS) - 0.425287f * LoadTexel(uvAndSampleIdx + uint3(1, 0, 0), w, h, nbS);
            T y_p1 = 0.425287f * LoadTexel(uvAndSampleIdx + uint3(-1, 1, 0), w, h, nbS) - 0.425287f * LoadTexel(uvAndSampleIdx + uint3(1, 1, 0), w, h, nbS);

            res += 0.229879f * y_m1 + 0.540242f * y_0 + 0.229879f * y_p1;
        } else if (mode == DerivativeMode.Optimal_5) {
            T y_m2 = 0.104550f * LoadTexel(uvAndSampleIdx + uint3(-2, -2, 0), w, h, nbS) + 0.292315f * LoadTexel(uvAndSampleIdx + uint3(-1, -2, 0), w, h, nbS) - 0.292315f * LoadTexel(uvAndSampleIdx + uint3(1, -2, 0), w, h, nbS) - 0.104550f * LoadTexel(uvAndSampleIdx + uint3(2, -2, 0), w, h, nbS);
            T y_m1 = 0.104550f * LoadTexel(uvAndSampleIdx + uint3(-2, -1, 0), w, h, nbS) + 0.292315f * LoadTexel(uvAndSampleIdx + uint3(-1, -1, 0), w, h, nbS) - 0.292315f * LoadTexel(uvAndSampleIdx + uint3(1, -1, 0), w, h, nbS) - 0.104550f * LoadTexel(uvAndSampleIdx + uint3(2, -1, 0), w, h, nbS);
            T y_0 = 0.104550f * LoadTexel(uvAndSampleIdx + uint3(-2, 0, 0), w, h, nbS) + 0.292315f * LoadTexel(uvAndSampleIdx + uint3(-1, 0, 0), w, h, nbS) - 0.292315f * LoadTexel(uvAndSampleIdx + uint3(1, 0, 0), w, h, nbS) - 0.104550f * LoadTexel(uvAndSampleIdx + uint3(2, 0, 0), w, h, nbS);
            T y_p1 = 0.104550f * LoadTexel(uvAndSampleIdx + uint3(-2, 1, 0), w, h, nbS) + 0.292315f * LoadTexel(uvAndSampleIdx + uint3(-1, 1, 0), w, h, nbS) - 0.292315f * LoadTexel(uvAndSampleIdx + uint3(1, 1, 0), w, h, nbS) - 0.104550f * LoadTexel(uvAndSampleIdx + uint3(2, 1, 0), w, h, nbS);
            T y_p2 = 0.104550f * LoadTexel(uvAndSampleIdx + uint3(-2, 2, 0), w, h, nbS) + 0.292315f * LoadTexel(uvAndSampleIdx + uint3(-1, 2, 0), w, h, nbS) - 0.292315f * LoadTexel(uvAndSampleIdx + uint3(1, 2, 0), w, h, nbS) - 0.104550f * LoadTexel(uvAndSampleIdx + uint3(2, 2, 0), w, h, nbS);

            res += 0.030320f * y_m2 + 0.249724f * y_m1 + 0.439911f * y_0 + 0.249724f * y_p1 + 0.030320f * y_p2;
        }

        return res;
    }

    float2 Get_DxDy(uint3 uvAndSampleIdx, const uint mode = DerivativeMode.Sobel) {
        if(N > 1){
            return float2(mul(T(1.0f), Get_ddx(uvAndSampleIdx, mode)), mul(T(1.0f), Get_ddy(uvAndSampleIdx, mode)));
        }
        else {
            return float2(Get_ddx(uvAndSampleIdx, mode)[0], Get_ddy(uvAndSampleIdx, mode)[0]);
        }
    }
};