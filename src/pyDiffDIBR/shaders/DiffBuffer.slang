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

struct CsteBuffer {
    //* Buffer containing the data in a sequential way
    //* We let the user access correctly the data through the argument 'position'
    TensorizedBuffer buff_;

    vector<float, N> load<let N : int>(uint position) {
        return buff_.load<N>(position);
    };

    matrix<float, R, C> load<let R : int, let C : int>(uint position) {
        return buff_.load<R, C>(position);
    }

    void save<let N : int>(uint position, vector<float, N> val) {
        buff_.save<N>(position, val);
    };

    void save<let R : int, let C : int>(uint position, matrix<float, R, C> val) {
        buff_.save<R, C>(position, val);
    };
};

struct DiffBuffer {
    //* Buffer containing the data in a sequential way
    //* We let the user access correctly the data through the argument 'position'
    TensorizedBuffer buff_;

    GradTensorizedBuffer grad_;

    [BackwardDerivative(bwd_load_v)]
    vector<float, N> load<let N : int>(uint position) {
        return buff_.load<N>(position);
    };

    void bwd_load_v<let N : int>(uint position, vector<float, N> val) {
        grad_.bwd_load_v<N>(position, val);
    }

    [BackwardDerivative(bwd_load_m)]
    matrix<float, R, C> load<let R : int, let C : int>(uint position) {
        return buff_.load<R, C>(position);
    }

    void bwd_load_m<let R : int, let C : int>(uint position, matrix<float, R, C> val) {
        grad_.bwd_load_m<R, C>(position, val);
    }

    void save<let N : int>(uint position, vector<float, N> val) {
        buff_.save<N>(position, val);
    };

    void save<let R : int, let C : int>(uint position, matrix<float, R, C> val) {
        buff_.save<R, C>(position, val);
    };

    vector<float, N> load_grad<let N : int>(uint position) { return grad_.load_grad<N>(position);};
    matrix<float, R, C> load_grad<let R : int, let C : int>(uint position) { return grad_.load_grad<R, C>(position);};
};

[shader("compute")]
[numthreads(16, 1, 1)]
void main_add_chunked_buffer(
    uint threadIdx: SV_DispatchThreadID,
    ConstantBuffer<ChunkedGradTensorizedBuffer> buffer
) {
    if (threadIdx >= buffer.nb_blocks.x) { return;}

    //[ForceUnroll]
    for (int i = 0; i < buffer.size; i++) {
        int total = 0;

        //[ForceUnroll]
        for (int idx_y = 0; idx_y < buffer.nb_blocks.y; idx_y++) {
            total += buffer.load(i, int2(threadIdx, idx_y));
        }
        buffer.set(i, int2(threadIdx, 0), total);
    }

    DeviceMemoryBarrier();

    if (threadIdx >= 1) { return;}

    //[ForceUnroll]
    for (int i = 0; i < buffer.size; i++) {
        int total = 0;

        //[ForceUnroll]
        for (int idx_x = 0; idx_x < buffer.nb_blocks.x; idx_x++) {
            total += buffer.load(i, int2(idx_x, 0));
        }
        buffer.set(i, int2(0, 0), total);
    }
};

struct ChunkedDiffBuffer {
    //* Buffer containing the data in a sequential way
    //* We let the user access correctly the data through the argument 'position'
    TensorizedBuffer buff_;

    ChunkedGradTensorizedBuffer grad_;

    [BackwardDerivative(bwd_load_v)]
    vector<float, N> load<let N : int>(uint position, int2 idx_block) {
        return buff_.load<N>(position);
    };

    void bwd_load_v<let N : int>(uint position, int2 idx_block, vector<float, N> val) {
        grad_.bwd_load_v<N>(position, idx_block, val);
    }

    [BackwardDerivative(bwd_load_m)]
    matrix<float, R, C> load<let R : int, let C : int>(uint position, int2 idx_block) {
        return buff_.load<R, C>(position);
    }

    void bwd_load_m<let R : int, let C : int>(uint position, int2 idx_block, matrix<float, R, C> val) {
        grad_.bwd_load_m<R, C>(position, idx_block, val);
    }

    void save<let N : int>(uint position, int2 idx_block, vector<float, N> val) {
        buff_.save<N>(position, val);
    };

    void save<let R : int, let C : int>(uint position, int2 idx_block, matrix<float, R, C> val) {
        buff_.save<R, C>(position, val);
    };

    vector<float, N> load_grad<let N : int>(uint position, int2 idx_block) { return grad_.load<N>(position, idx_block);};
    matrix<float, R, C> load_grad<let R : int, let C : int>(uint position, int2 idx_block) { return grad_.load<R, C>(position, idx_block);};

    int load_raw_grad(uint position, int2 idx_block) { return grad_.load(position, idx_block);};
    void set_raw_grad(uint position, int2 idx_block, int val) { grad_.set(position, idx_block, val);};
};

struct ChunkedGradTensorizedBuffer {
    //* Buffer containing the scaling values in a sequential way as the data buffer
    //* It is used to scale the gradient before saving it.
    //* We need it because we want to use Atomic<int32_t> for fast access,
    //* but we cannot guarantee that *incoming* gradients is ranged between 0,MAX_VALUE or -MAX_VALUE,MAX_VALUE so we need to scale it.
    //* But we let the user defines the right scaling based on the data.
    //* Rule of thumb: defines the max value that the parameter should have, then
    //* the scaling is `scaling = (2**16) / MAX_VALUE_PARAM`
    StructuredBuffer<float> grad_scaling;

    //* Buffer containing the gradient data in a sequential way as the data buffer.
    //* Atomic<int32_t> allows automatic Atomic operations and fast access.
    RWStructuredBuffer<Atomic<int32_t>> grad;

    //* Size of all buffers
    int size;

    //* Nb of blocks in X Y dimensions
    int2 nb_blocks; 

    //* Save computation if gradient is/must/should/will be ignored
    bool no_grad;

    int2 getIdxBlock(int2 pos, int2 max_pos) {
        return (pos * nb_blocks) / max_pos;
        //return int2(pos.x % nb_blocks.x, pos.y % nb_blocks.y);
    };

    int getPosBlock(int2 idx_block) {
        return size * idx_block.x + nb_blocks.x * idx_block.y * size;
    };

    void bwd_load_v<let N : int>(uint position, int2 idx_block, vector<float, N> val) {
        if (no_grad) { return;}
        
        int pos_block = getPosBlock(idx_block);
        if (position + N > size) { return;}

        [ForceUnroll]
        for (int i = 0; i < N; i++) {
            grad[pos_block + position + i] += int32_t(val[i] * grad_scaling[position + i]);
        }
    };

    void bwd_load_m<let R : int, let C : int>(uint position, int2 idx_block, matrix<float, R, C> val) {
        if (no_grad) { return;} 
        
        int pos_block = getPosBlock(idx_block);
        if (position + ((R - 1) * C + (C - 1)) > size) { return;}

        [ForceUnroll]
        for (int i = 0; i < R; i++) {
            [ForceUnroll]
            for (int j = 0; j < C; j++) {
                uint loc = i * C + j;
                grad[pos_block + loc + position] += int32_t(val[i][j] * grad_scaling[loc + position]);
            }
        }
    };

    vector<float, N> load<let N : int>(uint position, int2 idx_block) {
        vector<float, N> p = { 0.0f };

        int pos_block = getPosBlock(idx_block);
        if (position + N - 1 > size) { return p;}

        [ForceUnroll]
        for (int i = 0; i < N; i++) {
            p[i] = float(grad[pos_block + position + i].load()) / grad_scaling[position + i];
        }
        return p;
    };

    matrix<float, R, C> load<let R : int, let C : int>(uint position, int2 idx_block) {
        matrix<float, R, C> m = { 0.0f };

        int pos_block = getPosBlock(idx_block);
        if (position + ((R - 1) * C + (C - 1)) > size) { return m;}

        [ForceUnroll]
        for (int i = 0; i < R; i++) {
            [ForceUnroll]
            for (int j = 0; j < C; j++) {
                uint loc = i * C + j;
                m[i][j] = float(grad[pos_block + loc + position].load()) / grad_scaling[loc + position];
            }
        }
        return m;
    };

    int load(uint position, int2 idx_block) {
        int pos_block = getPosBlock(idx_block);
        if (position > size) { return 0;}

        return grad[pos_block + position].load();
    };

    void set(uint position, int2 idx_block, int val) {
        int pos_block = getPosBlock(idx_block);
        if (position > size) { return;}

        grad[pos_block + position].store(val);
    };
};

struct GradTensorizedBuffer {
    //* Buffer containing the scaling values in a sequential way as the data buffer
    //* It is used to scale the gradient before saving it.
    //* We need it because we want to use Atomic<int32_t> for fast access,
    //* but we cannot guarantee that *incoming* gradients is ranged between 0,MAX_VALUE or -MAX_VALUE,MAX_VALUE so we need to scale it.
    //* But we let the user defines the right scaling based on the data.
    //* Rule of thumb: defines the max value that the parameter should have, then
    //* the scaling is `scaling = (2**16) / MAX_VALUE_PARAM`
    StructuredBuffer<float> grad_scaling;

    //* Buffer containing the gradient data in a sequential way as the data buffer.
    //* Atomic<int32_t> allows automatic Atomic operations and fast access.
    RWStructuredBuffer<Atomic<int32_t>> grad;

    //* Size of all buffers
    int size;

    //* Save computation if gradient is/must/should/will be ignored
    bool no_grad;

    void bwd_load_v<let N : int>(uint position, vector<float, N> val) {
        if (no_grad) { return;}
        if (position + N > size) { return;}
        
        [ForceUnroll]
        for (int i = 0; i < N; i++) {
            grad[position + i] += int32_t(val[i] * grad_scaling[position + i]);
        }
    };
    
    void bwd_load_m<let R : int, let C : int>(uint position, matrix<float, R, C> val) {
        if (no_grad) { return;}
        if (position + ((R - 1) * C + (C - 1)) > size) { return ;}

        [ForceUnroll]
        for (int i = 0; i < R; i++) {
            [ForceUnroll]
            for (int j = 0; j < C; j++) {
                uint loc = i * C + j;
                grad[loc + position] += int32_t(val[i][j] * grad_scaling[loc + position]);
            }
        }
    };

    vector<float, N> load_grad<let N : int>(uint position) {
        vector<float, N> p = { 0.0f };
        if (position + N - 1 > size) { return p;}

        [ForceUnroll]
        for (int i = 0; i < N; i++) {
            p[i] = float(grad[position + i].load()) / grad_scaling[position + i];
        }
        return p;
    };

    matrix<float, R, C> load_grad<let R : int, let C : int>(uint position) {
        matrix<float, R, C> m = { 0.0f };
        if (position + ((R - 1) * C + (C - 1)) > size) { return m;}

        [ForceUnroll]
        for (int i = 0; i < R; i++) {
            [ForceUnroll]
            for (int j = 0; j < C; j++) {
                uint loc = i * C + j;
                m[i][j] = float(grad[loc + position].load()) / grad_scaling[loc + position];
            }
        }
        return m;
    };
};

struct TensorizedBuffer {
    //* Buffer containing the data in a sequential way
    //* We let the user access correctly the data through the argument 'position'
    RWStructuredBuffer<float> buffer;

    //* Size of all buffers
    int size;

    vector<float, N> load<let N : int>(uint position) {
        vector<float, N> p = { 0.0f };
        if (position + N > size) { return p;}

        [ForceUnroll]
        for (int i = 0; i < N; i++) {
            p[i] = buffer.Load(position + i);
        }
        return p;
    };

    matrix<float, R, C> load<let R : int, let C : int>(uint position) {
        matrix<float, R, C> m = { 0.0f };
        if (position + ((R - 1) * C + (C - 1)) > size) { return m;}

        [ForceUnroll]
        for (int i = 0; i < R; i++) {
            [ForceUnroll]
            for (int j = 0; j < C; j++) {
                uint loc = i * C + j;
                m[i][j] = buffer[loc + position];
            }
        }
        return m;
    }

    void save<let N : int>(uint position, vector<float, N> val) {
        if (position + N - 1 > size) { return;}

        [ForceUnroll]
        for (int i = 0; i < N; i++) {
            buffer[position + i] = val[i];
        }
    };
    
    void save<let R : int, let C : int>(uint position, matrix<float, R, C> val) {
        if (position + ((R - 1) * C + (C - 1)) > size) { return;}

        [ForceUnroll]
        for (int i = 0; i < R; i++) {
            [ForceUnroll]
            for (int j = 0; j < C; j++) {
                uint loc = i * C + j;
                buffer[loc + position] = val[i][j];
            }
        }
    };
};