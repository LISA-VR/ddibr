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

import DiffBuffer;

[shader("compute")]
[NumThreads(1,1,1)]
void single_thread_kernel_compute_R_t(
    uint threadIdx: SV_DispatchThreadID,
    uniform ConstantBuffer<DiffBuffer> param_cam_1,
    uniform ConstantBuffer<CsteBuffer> param_cam_2,
    uniform ConstantBuffer<DiffBuffer> R_t,
) {
    if (threadIdx > 1) { return;}

    float3x3 R;
    float3 t;
    single_thread_compute_R_t(param_cam_1, param_cam_2, R, t);

    R_t.save<3, 3>(0, R);
    R_t.save<3>(9, t);
};

[shader("compute")]
[NumThreads(1, 1, 1)]
void bwd_single_thread_kernel_compute_R_t(
    uint threadIdx: SV_DispatchThreadID,
    uniform ConstantBuffer<DiffBuffer> param_cam_1,
    uniform ConstantBuffer<CsteBuffer> param_cam_2,
    uniform ConstantBuffer<DiffBuffer> R_t,
) {
    if (threadIdx > 1) { return;}

    float3x3 dR = R_t.load_grad<3, 3>(0);
    float3 dt = R_t.load_grad<3>(9);
    bwd_diff(single_thread_compute_R_t)(param_cam_1, param_cam_2, dR, dt);
};

[BackwardDifferentiable]
void single_thread_compute_R_t(
    DiffBuffer param_cam_1,
    CsteBuffer param_cam_2,
    out float3x3 R,
    out float3 t,
) {
    float3 pos_1 = param_cam_1.load<3>(4);
    float3 rot_1 = param_cam_1.load<3>(7);

    float3 pos_2 = no_diff param_cam_2.load<3>(4);
    float3 rot_2 = no_diff param_cam_2.load<3>(7);

    R = SpaceTransformer.get_rotation_matrix_from_to(rot_1, rot_2);            
    t = SpaceTransformer.get_translation_matrix_from_to(pos_1, pos_2, rot_2);  
}

struct SpaceTransformer {

    [BackwardDifferentiable]
    static float3 get_translation_matrix_from_to(float3 pos_1, float3 pos_2, float3 rot_2) {
        float3x3 rot_mat_2 = get_rotation_matrix(rot_2);

        return mul(transpose(rot_mat_2), pos_1 - pos_2);
    };

    [BackwardDifferentiable]
    static float3x3 get_rotation_matrix_from_to(float3 rot_1, float3 rot_2) {
        float3x3 rot_mat_1 = get_rotation_matrix(rot_1);
        float3x3 rot_mat_2 = get_rotation_matrix(rot_2);

        return mul(transpose(rot_mat_2), rot_mat_1);
    };

    [BackwardDifferentiable]
    static float3x3 get_rotation_matrix(float3 rot) {
        float radperdeg = 0.01745329252f;

        return euler_angles_to_rot_matrix(radperdeg * rot); //* this gives the similar result to RVS
        //return transpose(eulerToMatrix(radperdeg * rot)); // equivalent with less matrix multiplication
    };

    [BackwardDifferentiable]
    static float3x3 eulerToMatrix(float3 euler) {
        float cx = cos(euler.z);
        float sx = sin(euler.z);
        float cy = cos(euler.y);
        float sy = sin(euler.y);
        float cz = cos(euler.x);
        float sz = sin(euler.x);

        return float3x3(
             cy * cz , cx * sz + sx * sy * cz, sx * sz - cx * sy * cz,
             -cy * sz, cx * cz - sx * sy * sz, sx * cz + cx * sy * sz,
             sy      , -sx * cy              , cx * cy
        );
    }

    [BackwardDifferentiable]
    static float3x3 euler_angles_to_rot_matrix(float3 rot) {
        float3x3 res = mul(rotation_matrix_from_rotation_around_y(rot[1]), rotation_matrix_from_rotation_around_x(rot[2]));
        res = mul(rotation_matrix_from_rotation_around_z(rot[0]), res);
        return res;
    };

    [BackwardDifferentiable]
    static float3x3 rotation_matrix_from_rotation_around_x(float rx) {
        return float3x3(
             float3(1.f, 0.F    , 0.f       ),
             float3(0.f, cos(rx), -sin(rx)  ),
             float3(0.f, sin(rx), cos(rx)   )
        );
    };

    [BackwardDifferentiable]
    static float3x3 rotation_matrix_from_rotation_around_y(float ry) {
        return float3x3(
            float3(cos(ry)  , 0.0f, sin(ry) ),
            float3(0.0f     , 1.0f, 0.0f    ),
            float3(-sin(ry) , 0.0f, cos(ry) )
        );
    };

    [BackwardDifferentiable]
    static float3x3 rotation_matrix_from_rotation_around_z(float rz) {
        return float3x3(
            float3(cos(rz), -sin(rz), 0.0f),
            float3(sin(rz),  cos(rz), 0.0f),
            float3(   0.0f,     0.0f, 1.0f)
        );
    };
};