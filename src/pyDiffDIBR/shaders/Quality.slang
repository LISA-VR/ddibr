/*
This file is based on/modified from "RVS software"
which is Copyright (c) 2010-2018 ITU/ISO/IEC. All rights reserved.

Original code is licensed under the BSD 3-Clause "New" or "Revised" License.
Modifications and Slang port are licensed under the GNU Affero General Public License v3.0.

For the BSD 3-Clause terms of the original work, see https://gitlab.com/mpeg-i-visual/rvs/-/blob/d757a353c7f84d228ac30c179d6ff9de356c6101/LICENSE.
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

import Uniforms;
import _Constants;
import _Utils;

[BackwardDifferentiable]
float Quality(
    Array<float4, 3> position,
    Array<float3, 3> input_position,
    Array<float3, 3> output_position,
    Array<float3, 3> uv,
    no_diff float2 size,
    no_diff float2 depth_range,
    no_diff QualityParameter qParams
) {
    #ifndef __TRAINING__
    if (
        ((depth_range[1] - input_position[0].x) * (depth_range[1] - input_position[0].x) <= cst_type_value<float>.EPS())  ||
        ((depth_range[0] - input_position[0].x) * (depth_range[0] - input_position[0].x) <= cst_type_value<float>.EPS())  
    ) {
        return 0.0f;
    }
    
    [ForceUnroll]
    for (int i = 0; i < 3; i++) {
    if (output_position[i].x <= 0.0f) {
        // Discard triangles behind the camera
        return 0.f;
    }}
    #endif

    float quality_foreground = quality_foreground(position[0], depth_range, qParams.scaling_power_quality_foreground);
    float quality_shape = quality_triangle_shape(position[0], position[1], position[2], size, qParams.scaling_power_quality_shape);
    float quality_normal = quality_triangle_normal(input_position[0], input_position[1], input_position[2], qParams.scaling_power_quality_normal);
    float quality_depth = quality_triangle_depth(input_position[0], input_position[1], input_position[2], depth_range, qParams.scaling_power_quality_depth);
    float quality_side = quality_triangle_side(position[0], position[1], position[2], size, qParams.scaling_power_quality_side);
    float quality_face_to_cam = quality_triangle_face_camera(output_position[0], output_position[1], output_position[2], qParams.scaling_power_quality_face_camera);

    constexpr const int nb_qualitys = 6;
    Array<float, nb_qualitys> qualitys = Array<float, nb_qualitys>(quality_shape, quality_normal, quality_depth, quality_side, quality_foreground, quality_face_to_cam);
    Array<float, nb_qualitys> scales = Array<float, nb_qualitys>(qParams.lambda_quality_shape, qParams.lambda_quality_normal, qParams.lambda_quality_depth, qParams.lambda_quality_side, qParams.lambda_quality_foreground, qParams.lambda_quality_face_camera);
    float quality = total_quality<nb_qualitys>(qualitys, scales);

    #ifndef __TRAINING__
    if (quality < qParams.quality_treshold) {
        return 0.0f;
    }
    #else
    float smooth_treshold = sigmoid<1>(quality - qParams.quality_treshold, 10.0f).x * (2.0f * qParams.quality_treshold);
    quality = min(smooth_treshold, quality);
    #endif
    return quality;
};

[BackwardDifferentiable]
float total_quality<let N : int>(
    Array<float, N> quality_list,
    no_diff Array<float, N> q_scale_list
) {
    float quality = cst_type_value<float>.EPS();
    float total_scale = cst_type_value<float>.EPS();

    [ForceUnroll]
    for (int i = 0; i < N; i++) {
        quality += q_scale_list[i] * quality_list[i];
        total_scale += q_scale_list[i];     
    }
    quality /= (total_scale);
    return quality;
}

[BackwardDifferentiable]
float quality_triangle_shape(
    float4 position_1,
    float4 position_2,
    float4 position_3,
    no_diff float2 size,
    no_diff float scaling_power = 1.0f
) { // based on triangle shape
    float2 A = position_1.xy * size;
    float2 B = position_2.xy * size;
    float2 C = position_3.xy * size; // Coordinate in [-w,w]x[-h,h]

    float dab = length(A - B);
    float dac = length(A - C);
    float dbc = length(B - C);

    float M = max(dbc, max(dab, dac));
    float m = min(dbc, min(dab, dac));

    float quality = max(0.0f, m / M - 0.3f) / (1.0f - 0.3f); // weirdo
    return pow(quality, scaling_power);
};

/**
 * Quality based on the normal
 * *Note: A,B,C are 3D points [z,x,y]
 */
[BackwardDifferentiable]
float quality_triangle_face_camera(
    float3 A,
    float3 B,
    float3 C,
    no_diff float scaling_power = 1.0f
) {
    if (A.x < 0.0f || C.x < 0.0f || B.x < 0.0f)
        return 0.0f;

    float3 normal = compute_normal(A, B, C);
    float3 n = float3(normal.x, normal.y, normal.z);
    float n_m = pow(dot(n * n, float3(1)), 0.5f);
    float v = (n.x) / n_m;
    return pow(max(0.f, v),scaling_power);
}

/**
* Quality based on the normal
* *Note: A,B,C are 3D points [z,x,y]
*/
[BackwardDifferentiable]
float quality_triangle_normal(
    float3 A,
    float3 B,
    float3 C,
    no_diff float scaling_power = 1.0f 
) { 
    if (A.x < 0.0f || C.x < 0.0f || B.x < 0.0f)
        return 0.0f;
    
    float3 normal = compute_normal(A, B, C);
    float3 n = float3(normal.x, normal.y, normal.z);
    float3 center = (A + B + C) / 3.0;
    // float v = dot(normalize(n),normalize(center));
    float v = dot(normalize(n), normalize(center));

    return pow(max(0.0f, abs(v) - 0.3f) / (1.0f - 0.3f), scaling_power);
}

/**
 * Quality based on the depth difference
 * *Note: A,B,C are 3D points [z,x,y]
 */
[BackwardDifferentiable]
float quality_triangle_depth(
    float3 A,
    float3 B,
    float3 C,
    no_diff float2 depth_range,
    no_diff float scaling_power = 1.0f 
) {
    if (A.x < 0.0 || C.x < 0.0 || B.x < 0.0)
        return 0;

    float dab = abs(A.x - B.x);
    float dac = abs(A.x - C.x);
    float dbc = abs(B.x - C.x);

    float depthdiff = max(dbc, max(dab, dac)) / depth_range[1];

    return pow(max(0.0f, (0.01f - depthdiff) / 0.01f), scaling_power);
}

/**
* Quality based on the triangle stretching 
*  ? (si ca strecth, c'est ca que ca ching?)
*  *Note : A, B, C are 3D points[z, x, y]
*/
[BackwardDifferentiable]
float quality_triangle_side(
    float4 position_1,
    float4 position_2,
    float4 position_3,
    no_diff float2 size, 
    no_diff float scaling_power = 1.0f
) { 
    float2 A = position_1.xy * size;
    float2 B = position_2.xy * size;
    float2 C = position_3.xy * size;

    float dab = length(A - B) / 2.0;
    float dac = length(A - C) / 2.0;
    float dbc = length(B - C) / 2.0;

    float stretch = max(dbc, max(dab, dac));

    float quality = max(0.0, 1.0f - (stretch / 15.0f));
    return pow(quality, scaling_power);
}

[BackwardDifferentiable]
float quality_foreground(float4 position, no_diff float2 depth_range, no_diff float scaling_power = 1.0f) {
    float q = 0.0f;
    //constexpr float EPS = cst_type_value<float>.EPS();
    //q = 1.f / (position.z + EPS);
    //float max_value = (1.f / depth_range[0]);
    //q = pow(min(max_value, max(0.f, q)) / max_value, scaling_power);
    q = pow(1.0 - position.z, scaling_power);
    return q;
};