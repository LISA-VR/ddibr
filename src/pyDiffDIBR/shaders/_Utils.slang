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

/**
 * ----------------------------------------------------------------------
 *  VERTEX BUFFER UTILS
 * ----------------------------------------------------------------------
*/

float2 getPositionFromVertexID(int id, int width) {
    int y = id / (width + 1);
    int x = id - y * (width + 1);
    return float2(float(x) - 0.5f, float(y) - 0.5f);
}

struct MeshBuffer {
    StructuredBuffer<uint>   indices;
    StructuredBuffer<float3> vertices;

    uint getVtxIdFromTriangleId(uint triangleId, uint vertexIdInTriangle) {
        uint vtxId = indices.Load(vertexIdInTriangle + triangleId * 3);
        return vtxId;
    }
    
    float3 getVtxPosFromTriangleId(uint triangleId, uint vertexIdInTriangle){
        uint vtxId = getVtxIdFromTriangleId(triangleId, vertexIdInTriangle);
        float3 vtxPos = vertices.Load(vtxId);
        return vtxPos;
    }
};

[BackwardDifferentiable]
float3 compute_normal(float3 vtx_1, float3 vtx_2, float3 vtx_3){
    float3 AB = vtx_2 - vtx_1;
    float3 AC = vtx_3 - vtx_1;

    float normal_x = AB.y * AC.z - AC.y * AB.z;
    float normal_y = AB.z * AC.x - AC.z * AB.x;
    float normal_z = AB.x * AC.y - AC.x * AB.y;
    return float3(normal_x, normal_y, normal_z);
}

/**
 * ----------------------------------------------------------------------
 *  UV/POS4 UTILS FUNCTIONS
 * ----------------------------------------------------------------------
*/

[BackwardDifferentiable]
float4 normalize_pos4(float4 pos, no_diff float2 imgSize) {
    return float4(pos.xy / imgSize * float2(2.0, -2.0) - float2(1.0, -1.0), pos.z, pos.w);
};

[BackwardDifferentiable]
vector<float, N> interpolate_vtx_attr<let N : int>(vector<float, N> uv_attr_1, vector<float, N> uv_attr_2, vector<float, N> uv_attr_3, float3 barycentricsCoords) {
    return barycentricsCoords.x * uv_attr_1 + barycentricsCoords.y * uv_attr_2 + barycentricsCoords.z * uv_attr_3;
};

[BackwardDifferentiable]
float4 uv_from_pos(float4 pos) {
    return float4((pos.xy / pos.w), pos.z / pos.w, pos.w / pos.w);
}

[BackwardDifferentiable]
float sarea_triangle(float2 a, float2 b, float2 c){
    float3x3 triangle = float3x3(
      a.x, a.y, 1.f,
      b.x, b.y, 1.f,
      c.x, c.y, 1.f
    );
    return 0.5f * determinant(triangle);
}


[BackwardDifferentiable]
float3 get_barycentrics_coords(no_diff float4 pos_pixel, float4 pos_vtx_1, float4 pos_vtx_2, float4 pos_vtx_3){
    
    float sarea_pbc = sarea_triangle(pos_pixel.xy, pos_vtx_2.xy, pos_vtx_3.xy);
    float sarea_abc = sarea_triangle(pos_vtx_1.xy, pos_vtx_2.xy, pos_vtx_3.xy);
    float sarea_apc = sarea_triangle(pos_vtx_1.xy, pos_pixel.xy, pos_vtx_3.xy);
    float sarea_abp = sarea_triangle(pos_vtx_1.xy, pos_vtx_2.xy, pos_pixel.xy);
    
    float lambda_1 = sarea_pbc / sarea_abc;
    float lambda_2 = sarea_apc / sarea_abc;
    float lambda_3 = sarea_abp / sarea_abc;
    
    return float3(lambda_1, lambda_2, lambda_3);
}

/**
 * ----------------------------------------------------------------------
 *  SIGMOID UTILS FUNCTIONS
 * ----------------------------------------------------------------------
*/

[BackwardDifferentiable]
vector<float, N> sigmoid<let N : int>(vector<float, N> a, no_diff float lambda) {
    return 1.f / (1.f + exp(-lambda * a));
};

[BackwardDerivativeOf(sigmoid)]
void bwd_sigmoid<let N : int>(inout DifferentialPair<vector<float, N>> d_a, float lambda, vector<float, N> grad) {
    let y = sigmoid(d_a.p, lambda);
    let d_y = lambda * y * (1.f - y);
    d_a = diffPair(d_a.p, d_y * grad);
};

[BackwardDifferentiable]
vector<float, N> sigmoid_shift_scale<let N : int>(vector<float, N> a, no_diff float lambda, no_diff float shift, no_diff float scale) {
    return (sigmoid(a, lambda) * scale) + shift;
};

[BackwardDifferentiable]
vector<float, N> gelu<let N : int>(vector<float, N> x) {
    return 0.5f * (x) * (1.f + tanh(pow(2.f / 3.14f, 0.5f) * (x + 0.044715f * pow(x, 3.f))));
}
