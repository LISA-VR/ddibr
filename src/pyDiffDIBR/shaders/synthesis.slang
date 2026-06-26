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

import Projection;
import Uniforms;
import ShaderStageOut;
import DiffTex;
import WorldTransformation;
import _Utils;
import Quality;
import _Constants;

/**
 * ------------------------------------------------------------------------------
 * VERTEX SHADER INSPIRED BY RVS CODE SOURCE
 
    .. references::
    [1] *'Bonatto, D., Fachada, S., Lafruit, G.: RaViS: Real-time accelerated View Synthesizer for immersive video 6DoF VR. Electronic Imaging pp. 382–1 (01 2020)'*
    [2] *'Bonatto, D., Fachada, S., Rogge, S., Munteanu, A., Lafruit, G.: Real-Time Depth Video Based Rendering for 6-DoF HMD Navigation and Light Field Displays. IEEE Access 9, 1–1 (01 2021)'*
    [3] *'Dury, S., Bonatto, D., Teratani, M., & Lafruit, G. (2022). 3D Computer Graphics: View Synthesis Tool for VR Immersive Video.'*
 * ------------------------------------------------------------------------------
*/

[shader("vertex")]
VertexShader_OUT vertex_main(
    float3 vtx_position : POSITION, 
    uniform ConstantBuffer<SpaceTransform> transform,
    uniform ConstantBuffer<DiffCamera> in_diff_cam,
    uniform ConstantBuffer<CameraParameters> out_cam_params,
)
{
    float2 xy  = vtx_position.xy;
    float3 xyz = loadXYZ(in_diff_cam.depth_texture, in_diff_cam.sampler_tex, xy, in_diff_cam.params);
    float3 eucl = unproject_xy_to_world(xyz, in_diff_cam.params); // {z, x, y}
    float3 eucl_ = rotate_translate_world(eucl, transform);
    float4 pos = project_xyz_to_camera(eucl_, out_cam_params);

    VertexShader_OUT vs_out;
    vs_out.uv = xyz / float3(in_diff_cam.params.imgSize, 1.0);
    vs_out.position = normalize_pos4(pos, float2(out_cam_params.imgSize));

    vs_out.position.z = clamp(vs_out.position.z, out_cam_params.min_max_depth[0], out_cam_params.min_max_depth[1]);
    vs_out.position.z = (vs_out.position.z - out_cam_params.min_max_depth[0]) / (out_cam_params.min_max_depth[1] - out_cam_params.min_max_depth[0]);
    vs_out.position.z = clamp(vs_out.position.z, 0.001f, 0.999f);

    vs_out.depth_range = in_diff_cam.params.min_max_depth;
    vs_out.size_img = float2(out_cam_params.imgSize);
    vs_out.input_pos = eucl;
    vs_out.output_pos = eucl_;

    return vs_out;
};

/**
 * ------------------------------------------------------------------------------
 * GEOMETRY SHADER INSPIRED BY RVS CODE SOURCE
 
 .. references::
 [1] *'Bonatto, D., Fachada, S., Lafruit, G.: RaViS: Real-time accelerated View Synthesizer for immersive video 6DoF VR. Electronic Imaging pp. 382–1 (01 2020)'*
 [2] *'Bonatto, D., Fachada, S., Rogge, S., Munteanu, A., Lafruit, G.: Real-Time Depth Video Based Rendering for 6-DoF HMD Navigation and Light Field Displays. IEEE Access 9, 1–1 (01 2021)'*
 [3] *'Dury, S., Bonatto, D., Teratani, M., & Lafruit, G. (2022). 3D Computer Graphics: View Synthesis Tool for VR Immersive Video.'*
 * ------------------------------------------------------------------------------
*/

void emit_vertex(
    VertexShader_OUT vertex,
    TriangleStream<GeometryShader_OUT> outputStream,
    float  quality,
    float normal,
    uint primitiveID 
) {
    GeometryShader_OUT out_vertex;
    out_vertex.position = vertex.position;
    out_vertex.uv = vertex.uv;
    out_vertex.quality = quality;
    out_vertex.primitiveID = primitiveID;
    out_vertex.normal = normal;

    // Process and append vertices
    outputStream.Append(out_vertex);
}

[shader("geometry")]
[maxvertexcount(3)]
void geometry_main(
    triangle VertexShader_OUT vertex[3],
    inout TriangleStream<GeometryShader_OUT> outputStream,
    uniform ConstantBuffer<QualityParameter> qParams,
    uint primitiveID: SV_PrimitiveID
) {
    float quality = Quality(
        Array<float4, 3>(vertex[0].position, vertex[1].position, vertex[2].position),
        Array<float3, 3>(vertex[0].input_pos, vertex[1].input_pos, vertex[2].input_pos),
        Array<float3, 3>(vertex[0].output_pos, vertex[1].output_pos, vertex[2].output_pos),
        Array<float3, 3>(vertex[0].uv, vertex[1].uv, vertex[2].uv),
        vertex[0].size_img, vertex[0].depth_range,
        qParams
    );

    #ifndef __TRAINING__
    if (quality <= 0.0f) { return;} //* do not emit vertices if quality is bad and not during training
    #endif
    
    float3 normal = compute_normal(vertex[0].output_pos, vertex[1].output_pos, vertex[2].output_pos);
    float3 n = float3(normal.x, normal.y, normal.z);
    float n_m = pow(dot(n * n, float3(1)), 0.5f); //todo
    float v = (n.x) / n_m;
    
    emit_vertex(vertex[0], outputStream, quality, v, primitiveID);
    emit_vertex(vertex[1], outputStream, quality, v, primitiveID);
    emit_vertex(vertex[2], outputStream, quality, v, primitiveID);
    outputStream.RestartStrip();
}


/**
 * ------------------------------------------------------------------------------
 * FRAGMENT SHADER INSPIRED BY RVS CODE SOURCE
 
 .. references::
 [1] *'Bonatto, D., Fachada, S., Lafruit, G.: RaViS: Real-time accelerated View Synthesizer for immersive video 6DoF VR. Electronic Imaging pp. 382–1 (01 2020)'*
 [2] *'Bonatto, D., Fachada, S., Rogge, S., Munteanu, A., Lafruit, G.: Real-Time Depth Video Based Rendering for 6-DoF HMD Navigation and Light Field Displays. IEEE Access 9, 1–1 (01 2021)'*
 [3] *'Dury, S., Bonatto, D., Teratani, M., & Lafruit, G. (2022). 3D Computer Graphics: View Synthesis Tool for VR Immersive Video.'*
 * ------------------------------------------------------------------------------
*/

[BackwardDifferentiable]
float4 shadeFragment(DifferentiableTexture<4> tex, no_diff SamplerState s, no_diff float2 uv) {
    float4 color = tex.sample(s, uv);
    color = sigmoid(color, 1.0);
    return color;
};

[shader("fragment")]
FragmentShader_OUT fragment_main(
    GeometryShader_OUT input,
    uniform ConstantBuffer<DiffCamera> in_diff_cam,
)
{
    float2 uv = input.uv.xy;
    float4 color = shadeFragment(in_diff_cam.color_texture, in_diff_cam.sampler_tex, uv);
    float depth = input.position.z;

    FragmentShader_OUT frag_out;
    frag_out.frag_color = color;
    frag_out.frag_depthmap = depth;
    frag_out.frag_quality = input.quality;
    frag_out.frag_mask = 1.0f;
    frag_out.frag_normal = input.normal;

    return frag_out;
};

/**
 * ------------------------------------------------------------------------------
 * BWD FRAGMENT SHADER
 * ------------------------------------------------------------------------------
*/

float4 getReferenceFragment(no_diff CstTexture<4> tex, no_diff SamplerState s, no_diff float2 uv) {
    return tex.sample(s, uv);
};

[shader("fragment")]
void diff_fragment_main(
    GeometryShader_OUT input,
    uniform ConstantBuffer<DiffCamera> in_diff_cam,
    uniform ConstantBuffer<Camera> target_cam,
    uniform ConstantBuffer<SpaceTransform> transform,
    uniform ConstantBuffer<MeshBuffer> meshBuffer,
    uniform ConstantBuffer<CopyDrawResult> drawResult,
    uniform ConstantBuffer<BlendingResult> blendingResult,
    uniform ConstantBuffer<OptimizationParameters> opti_param,
    uniform ConstantBuffer<QualityParameter> qParams,
    float3 barycentric_coords : SV_BaryCentrics,
    uint primitiveID: SV_PrimitiveID,
    // uint sample_index : SV_SampleIndex //TODO: SampleIdx cannot work for now since the Sampling Location is not *necesserly* fixed... https://github.com/shader-slang/slang-rhi/blob/main/src/vulkan/vk-command.cpp#L786
)
{
    // ---------------------------- Find right sampled color ----------------------------
    //* Discard fragment if the interpolated depth is too different from the rendered depth buffer.
    #ifndef NB_SAMPLES
    #define NB_SAMPLES 1
    #endif

    bool incorrectDepth = true;
    uint w; uint h; uint nbS;
    int closestSampleIdx = 0;
    float bestScore = 1e5;
    drawResult.ms_depth.GetDimensions(w, h, nbS);
    [ForceUnroll]
    for (int sIdx = 0; sIdx < NB_SAMPLES; sIdx++) {
        float d     = drawResult.ms_depth.Load(int2(input.position.xy), sIdx).x;
        float d_    = input.position.z;
        float se = (d_ - d) * (d_ - d);
        if (se < opti_param.depth_check_tol) {
            incorrectDepth = false;
            if (se < bestScore) {
                bestScore = se;
                closestSampleIdx = sIdx;
            }
        }
    }
    if (incorrectDepth) return;

    float2 render_uv = input.uv.xy;
    float2 ref_uv = (input.position.xy) / float2(target_cam.params.imgSize);
    uint3 uv_sample = uint3(int2(input.position.xy), closestSampleIdx);

    // ---------------------------- Diff Multi-Sampled $\partial C / \partial c_s = c_s / S$ ----------------------------
    float4 dL_dCs   = drawResult.color.LoadGrad(int2(input.position.xy), target_cam.params.imgSize[0]) / float(NB_SAMPLES);
    float dL_dQs = drawResult.quality.LoadGrad(int2(input.position.xy), target_cam.params.imgSize[0]).x / float(NB_SAMPLES);
    float dL_dMs = drawResult.mask.LoadGrad(int2(input.position.xy), target_cam.params.imgSize[0]).x / float(NB_SAMPLES);
    float dL_dDs = drawResult.depth.LoadGrad(int2(input.position.xy), target_cam.params.imgSize[0]).x / float(NB_SAMPLES);

    // ------------------- Diff $\partial c_s / \partial c_i = \partial c_i(Warping) / \partial c_i$  -------------------
    bwd_diff(shadeFragment)(in_diff_cam.color_texture, in_diff_cam.sampler_tex, render_uv, dL_dCs * float4(1.f, 1.f, 1.f, 1.f));

    // ------------------- Diff $\partial c_s / \partial \{d_i, ...\} = \partial c_s / \partial \{u,v\} \partial \{u,v\} / \partial \{d_i, ...\}$  -------------------
    MSCstTexture<4> color_ms = MSCstTexture<4>(drawResult.ms_color);
    float2 dCs_dAdY = color_ms.Get_DxDy(uv_sample, DerivativeMode.Optimal_3); // $\{\partial c^i_{out}/\partial u,\partial c^i_{out}/\partial v \}$

    // ------------------- Diff $\partial q_s / \partial \{d_i, ...\} = \partial q_s / \partial \{u,v\} \partial \{u,v\} / \partial \{d_i, ...\}$  -------------------
    MSCstTexture<1> quality_ms = MSCstTexture<1>(drawResult.ms_quality);
    float2 dQs_dAdY = quality_ms.Get_DxDy(uv_sample, DerivativeMode.Optimal_3); // $\partial q_{out} / \partial\{u,v\}$

    MSCstTexture<1> depth_ms = MSCstTexture<1>(drawResult.ms_depth);
    float2 dDs_dAdY = depth_ms.Get_DxDy(uv_sample, DerivativeMode.Optimal_3); // $\partial q_{out} / \partial\{u,v\}$

    MSCstTexture<1> mask_ms = MSCstTexture<1>(drawResult.ms_mask);
    float2 dMs_dAdY = mask_ms.Get_DxDy(uv_sample, DerivativeMode.Optimal_3); // $\partial q_{out} / \partial\{u,v\}$

    float4 dL_dPos = float4(
        mul(1.f, dL_dCs) * dCs_dAdY.x + dL_dQs * dQs_dAdY.x + dL_dDs * dDs_dAdY.x + dL_dMs * dMs_dAdY.x,
        mul(1.f, dL_dCs) * dCs_dAdY.y + dL_dQs * dQs_dAdY.y + dL_dDs * dDs_dAdY.y + dL_dMs * dMs_dAdY.y,
        dL_dDs,                                              
        0.0f,
    );

    GeometryShader_OUT_mimick gOut_diff;
    gOut_diff.position   = dL_dPos;
    gOut_diff.quality    = dL_dQs;
    gOut_diff.uv         = 0.f;

    bwd_diff(vertex_geometry_shader_mimick)(primitiveID, in_diff_cam, target_cam, transform, meshBuffer, barycentric_coords, qParams, gOut_diff);

    return;
};

struct GeometryShader_OUT_mimick : IDifferentiable {
    float3 uv;
    float4 position;
    float quality;
}

[BackwardDifferentiable]
GeometryShader_OUT_mimick vertex_geometry_shader_mimick(
    no_diff uint primitiveID,
    DiffCamera in_diff_cam,
    no_diff Camera target_cam,
    no_diff SpaceTransform transform,
    no_diff MeshBuffer meshBuffer,
    no_diff float3 baryCentricsCoords,
    no_diff QualityParameter qParams,
) {
    Array<float4, 3> pos4_list;
    Array<float3, 3> in_pos_list;
    Array<float3, 3> out_pos_list;
    Array<float3, 3> uv_list;

    [ForceUnroll]
    for (int vtx = 0; vtx < 3; vtx++){
        //uint vtxId = no_diff idxBuff.getVtxIdFromTriangleId(primitiveID, vtx);
        //float2 xy = no_diff getPositionFromVertexID(vtxId, in_diff_cam.params.imgSize[0]);
        float3 vtx_pos = no_diff meshBuffer.getVtxPosFromTriangleId(primitiveID, vtx);
        float2 xy = vtx_pos.xy;
        int2 idx_block = transform.R_t.grad_.getIdxBlock(int2(xy) + vtx * int2(target_cam.params.imgSize), int2(target_cam.params.imgSize) * 3);

        float3 xyz = loadXYZ(in_diff_cam.depth_texture, in_diff_cam.sampler_tex, xy, in_diff_cam.params);
        float3 eucl = unproject_xy_to_world(xyz, in_diff_cam.params, idx_block);
        float3 eucl_ = rotate_translate_world(eucl, transform, idx_block);
        float4 xyzw = project_xyz_to_camera(eucl_, target_cam.params);
        pos4_list[vtx] = xyzw;

        pos4_list[vtx].z = clamp(pos4_list[vtx].z, target_cam.params.min_max_depth[0], target_cam.params.min_max_depth[1]);
        pos4_list[vtx].z = (pos4_list[vtx].z - target_cam.params.min_max_depth[0]) / (target_cam.params.min_max_depth[1] - target_cam.params.min_max_depth[0]);
        pos4_list[vtx].z = clamp(pos4_list[vtx].z, 0.001f, 0.999f);
        
        pos4_list[vtx] = uv_from_pos(pos4_list[vtx]);
        in_pos_list[vtx] = eucl;
        out_pos_list[vtx] = eucl_;
        uv_list[vtx] = xyz / float3(in_diff_cam.params.imgSize, 1.f);
    }

    float quality = Quality(
        pos4_list,
        in_pos_list,
        out_pos_list,
        uv_list,
        float2(in_diff_cam.params.imgSize), 
        in_diff_cam.params.min_max_depth, 
        qParams
    );

    float4 pos4         = interpolate_vtx_attr<4>(pos4_list[0], pos4_list[1], pos4_list[2], baryCentricsCoords);
    float3 uv           = interpolate_vtx_attr<3>(uv_list[0], uv_list[1], uv_list[2], baryCentricsCoords);

    GeometryShader_OUT_mimick gOut;
    gOut.position   = pos4;
    gOut.quality    = quality;
    gOut.uv         = uv;

    return gOut;
};