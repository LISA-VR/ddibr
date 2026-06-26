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

void basic_mean_texture<let N : int, let NB_x : int, let NB_y : int>(
    uint3 id,
    uint2 textureSize,
    Texture2D<vector<float, N>> inputTexture,
    RWStructuredBuffer<vector<float, N>> sum_buffer
) {
    typedef vector<float, N> _VFloat;

    constexpr int2 nb_blocks = int2(NB_x, NB_y);
    int2 nb_pixel_per_block = int2((textureSize.x + nb_blocks.x) / nb_blocks.x, (textureSize.y + nb_blocks.y) / nb_blocks.y);

    int group_id = id.x + id.y * nb_blocks.x;
    if (group_id >= nb_blocks.x * nb_blocks.y) {return;}

    int2 location = int2(
        nb_pixel_per_block.x * id.x,
        nb_pixel_per_block.y * id.y
    );

    _VFloat sum = _VFloat(0.f);
    for (int x = 0; x < nb_pixel_per_block.x; x++) {
        int x_ = location.x + x;
        if ((x_ < 0) || (x_ >= textureSize.x)) { break;}
        //[ForceUnroll]
        for (int y = 0; y < nb_pixel_per_block.y; y++) {
            int y_ = location.y + y;
            if ((y_ < 0) || (y_ >= textureSize.y)) { break;}

            sum += inputTexture.Load(int3(x_, y_, 0));
        }
    }

    constexpr float channels = (N != 4) ? float(N) : float(N - 1);
    [ForceUnroll]
    for (int i = 0; i < channels; i++) {
        sum_buffer[group_id][i] = sum[i] / (textureSize.x * textureSize.y * channels);
    }
}

// Define the workgroup size
[shader("compute")]
[NumThreads(16, 16, 1)]
void main_basic_mean_texture_t4(
    uint3 threadId: SV_DispatchThreadID,
    uint3 groupId: SV_GroupID,
    uint groupIndex: SV_GroupIndex,
    uint3 groupThreadId : SV_GroupThreadID,
    uniform Texture2D<float4> inputTexture,
    uniform RWStructuredBuffer<float4> outputBuffer,
)
{
    // Determine the size of the texture
    uint2 textureSize;
    inputTexture.GetDimensions(textureSize.x, textureSize.y);

    basic_mean_texture<4, 64, 64>(threadId, textureSize, inputTexture, outputBuffer);
}
