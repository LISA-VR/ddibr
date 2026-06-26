"""
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
"""

import numpy as np
import fast_simplification
import slangpy as spy

from .RenderingDevice import RenderingDevice
from ._utils import Logger

class MeshBuffer:
    
    def __init__(self, vtx_buffer : np.ndarray, idx_buffer : np.ndarray, label : str = "none"):
        self._vtx_buffer_cpu : np.ndarray = vtx_buffer
        self._idx_buffer_cpu : np.ndarray = idx_buffer
        self._vtx_buffer_gpu : spy.Buffer
        self._idx_buffer_gpu : spy.Buffer
        self.label = label
        
        self._rd = RenderingDevice()
        self.update_gpu()
    
    def update_gpu(self):
        self._vtx_buffer_gpu = self._rd.create_vertex(self._vtx_buffer_cpu, f"vtx_buffer_{self.label}")
        self._idx_buffer_gpu = self._rd.create_index(self._idx_buffer_cpu, f"idx_buffer_{self.label}")
    
    @property
    def vtx_arr(self) -> np.ndarray:
        return self._vtx_buffer_cpu
        
    @property
    def idx_arr(self) -> np.ndarray:
        return self._idx_buffer_cpu
    
    @property
    def vtx_buffer(self) -> spy.Buffer:
        return self._vtx_buffer_gpu
        
    @property
    def idx_buffer(self) -> spy.Buffer:
        return self._idx_buffer_gpu
        
    
class SimpleFlatMeshBuffer(MeshBuffer):
    """
    Similar class to VAO_VBO_EBO in RVS.cpp
    
    .. references::
    [1] *'Bonatto, D., Fachada, S., Lafruit, G.: RaViS: Real-time accelerated View Synthesizer for immersive video 6DoF VR. Electronic Imaging pp. 382–1 (01 2020)'*
    [2] *'Bonatto, D., Fachada, S., Rogge, S., Munteanu, A., Lafruit, G.: Real-Time Depth Video Based Rendering for 6-DoF HMD Navigation and Light Field Displays. IEEE Access 9, 1–1 (01 2021)'*
    [3] *'Dury, S., Bonatto, D., Teratani, M., & Lafruit, G. (2022). 3D Computer Graphics: View Synthesis Tool for VR Immersive Video.'*
    """
    
    
    _NB_VERTEX_COORDINATES = 3
    _NB_TRIANGLE_BY_PIXEL  = 2

    def __init__(self, width : int ,height : int, label : str = "none"):
        self.width = width
        self.height = height

        #self.vao = None #not useful anymore
        vertex_buffer = SimpleFlatMeshBuffer.create_vertex_buffer_from_image(width, height)
        index_buffer = SimpleFlatMeshBuffer.create_index_buffer_from_image(width, height)
        
        super().__init__(vertex_buffer, index_buffer, label=label)

    @staticmethod
    def create_index_buffer_from_image(width : int, height : int, dtype : str = "uint32") -> np.ndarray:
        el_count = SimpleFlatMeshBuffer._NB_VERTEX_COORDINATES * SimpleFlatMeshBuffer._NB_TRIANGLE_BY_PIXEL * (height) * (width)
        
        index_buff = np.empty(el_count, dtype=dtype)

        # Create arrays of indices
        y_indices = np.arange(height)
        x_indices = np.arange(width)
        
        # Use broadcasting to compute all indices
        Y, X = np.meshgrid(y_indices, x_indices, indexing='ij')
        
        # Base index in expanded grid
        base = Y * (width + 1) + X
        
        # Fill the index buffer using stride tricks
        index_buff[0::6] = base.ravel()                    # (x, y)
        index_buff[1::6] = (base + 1).ravel()              # (x+1, y)
        index_buff[2::6] = (base + width + 1).ravel()      # (x, y+1)
        index_buff[3::6] = (base + width + 1).ravel()      # (x, y+1)
        index_buff[4::6] = (base + 1).ravel()              # (x+1, y)
        index_buff[5::6] = (base + width + 2).ravel()      # (x+1, y+1)

        return index_buff
    
    @staticmethod
    def create_vertex_buffer_from_image(width : int, height : int, dtype : str = "float32") -> np.ndarray:
        x_coords = np.arange(width + 1, dtype=np.float32)
        y_coords = np.arange(height + 1, dtype=np.float32)
        
        # Create meshgrid of all vertex positions
        x, y = np.meshgrid(x_coords, y_coords)
        
        # Flatten to get vertex positions (x, y, 0) - z=0 for flat mesh
        # If you want to use pixel intensity as height, modify this
        vertices = np.column_stack([
            x.flatten(),
            y.flatten(),
            np.zeros((height + 1) * (width + 1), dtype=np.float32)
        ]).astype(dtype)
        return vertices
        

class SimplifiableMeshBuffer(SimpleFlatMeshBuffer):
    def __init__(self, width : int ,height : int, label : str = "none"):
        self._simplified_vtx_buffer_cpu : np.ndarray = np.zeros((width * height), dtype="float32") # tmp array
        self._simplified_idx_buffer_cpu : np.ndarray = np.zeros((width * height), dtype="float32") # tmp array
        super().__init__(width, height, label=label)
        
        self._simplified_vtx_buffer_cpu = self._vtx_buffer_cpu.copy()
        self._simplified_idx_buffer_cpu = self._idx_buffer_cpu.copy()
        self._vtx_buffer_gpu : spy.Buffer
        self.update_gpu()
        
    def simplify(self, depth : np.ndarray, mesh_reduction : float = 0.9):
        z = depth.flatten()
        vertices = self._vtx_buffer_cpu.copy()
        vertices[:z.shape[0], 2] = z
        faces = self.idx_arr.reshape((-1, 3))
        new_vertices, new_indices = fast_simplification.simplify(vertices, faces, mesh_reduction)[0:2]
        self._simplified_vtx_buffer_cpu, self._simplified_idx_buffer_cpu = new_vertices.astype(vertices.dtype), new_indices.reshape((-1,)).astype(faces.dtype)
        
        Logger.debug(f"Mesh {self.label} has been reduced by a factor: {faces.shape[0] / new_indices.shape[0] : .2f}")
        
        self.update_gpu()
    
    def update_gpu(self):
        self._vtx_buffer_gpu = self._rd.create_vertex(self._simplified_vtx_buffer_cpu, f"simplified_vtx_buffer_{self.label}")
        self._idx_buffer_gpu = self._rd.create_index(self._simplified_idx_buffer_cpu, f"simplified_idx_buffer_{self.label}")