<p align="center">
  <h1 align="center">&#8706;DIBR: Differentiable Depth Image-based Rendering for Fast Novel View Synthesis </h1>
  <p align="center">
    <a href="https://orcid.org/0000-0001-6923-3206">Armand Losfeld</a>
    &nbsp;·&nbsp;
    <a href="https://orcid.org/0000-0003-1667-8177">Sarah Dury</a>
    &nbsp;·&nbsp;
    <a href="https://orcid.org/0000-0002-2349-1936">Gauthier Lafruit</a>
    &nbsp;·&nbsp;
    <a href="https://orcid.org/0000-0001-9332-1409">Mehrdad Teratani</a>
    &nbsp;·&nbsp;
    <a href="https://orcid.org/0000-0001-6502-5354">Daniele Bonatto</a>
  </p>
  <h3 align="center">ECCV 2026 Poster</h3>
  <h3 align="center"><a href="https://arxiv.org/abs/2403.14627">Paper</a> | <a href="https://donydchen.github.io/mvsplat/">Project Page</a></h3>
    <!-- 
    <ul>
    <li><b href="https://github.com/donydchen/mvsplat360">Update:</b> Fix, etc.</li>
    <li><b href="https://github.com/donydchen/mvsplat360">Original Implementation</b></li> 
    </ul>
    -->
</p>

---

## Cite this work

```
@inbook{diffdibr_eccv_2026,
  title={$\partial$DIBR: Differentiable Depth Image-based Rendering for Fast Novel View Synthesis},
  author={Losfeld, Armand and Dury, Sarah and Lafruit, Gauthier and Teratani, Mehrdad and Bonatto, Daniele},
  booktitle={Computer Vision – ECCV 2026}, 
  publisher={Springer Nature}, 
  year={2026},
}
```

---

## Table of Contents
- [1. System Requirements](#1-system-requirements)
  * [Recommended Environment](#recommended-environment)
  * [Download the repo](#download-the-repo)
- [2. Installing NerfBaselines](#2-installing-nerfbaselines)
  * [2.1 Create the Conda Environment](#21-create-the-conda-environment)
  * [2.2 Install the API](#22-install-the-api)
- [3. Installing Rendering Methods with Nerfbaselines](#3-installing-rendering-methods-with-nerfbaselines)
- [4. Dataset Download and Preparation](#4-dataset-download-and-preparation)
  * [4.1 LLFF Dataset](#41-llff-dataset)
  * [4.2 Shiny Dataset](#42-shiny-dataset)
- [5. Training Ours and Baselines Methods](#5-training-ours-and-baselines-methods)
  * [5.1 Default Training](#51-default-training)
  * [5.2 Our Coarse-to-Fine Optimization](#52-our-coarse-to-fine-optimization)
    + [5.2.1 Optional Parameters](#521-optional-parameters)
- [7. Interactive Viewer](#7-interactive-viewer)
- [8. Results](#8-results)
- [9. Standalone Installation Without NerfBaselines](#9-standalone-installation-without-nerfbaselines)
  * [9.1 Installation](#91-installation)
    + [9.1.1 Depth Submodules](#911-depth-submodules)
      - [9.1.1.1 ZoeDepth Installation](#9111-zoedepth-installation)
      - [9.1.1.2 DepthAnything3 Installation](#9112-depthanything3-installation)
  * [9.2 Optimization](#92-optimization)
  * [9.3 Standalone Viewer](#93-standalone-viewer)
- [10. Uninstallation](#10-uninstallation)
  * [10.1 Remove conda environment](#101-remove-conda-environment)
  * [10.2 Delete Nerfbaselines folder](#102-delete-nerfbaselines-folder)
  * [10.3 Optional: Cleaning conda cache](#103-optional--cleaning-conda-cache)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Dataset Licenses](#dataset-licenses)

---

## 1. System Requirements

### Recommended Environment
- **Linux** (e.g., Ubuntu 24.04) — strongly recommended 
- **NVIDIA RTX GPU**, Vulkan + Slang support, 10–24 GiB VRAM 
- [**CUDA**](https://developer.nvidia.com/cuda-downloads) (version matching your NVIDIA driver) 
- [**Conda**](https://anaconda.org/) (Anaconda or Miniconda) 
- [**apptainer**](https://apptainer.org/) (for containerized installations)

> [!NOTE]
  Some methods require GPU‑based Vulkan rendering; ensure proper driver installation. On laptops, explicitly ensure the NVIDIA GPU is used.

### Download the repo

For _nerfbaseline-integrated_ installation, 

```bash
git clone https://github.com/LISA-VR/ddbir.git
git submodule update --init --remote external/nerfbaselines
```

For standalone installation,

```bash
git clone https://github.com/LISA-VR/ddbir.git
git submodule update --init --remote external/ZoeDepth
git submodule update --init --remote external/DepthAnything3 #Optional
```

---

## 2. Installing NerfBaselines

All evaluated methods (ours and baselines) are integrated into a unified evaluation workflow using the [**NerfBaselines**](https://nerfbaselines.github.io/) API. A *modified fork* containing our integration is provided in [`external/nerfbaselines`](external/nerfbaselines-fork/). The forked project can also be found at [https://github.com/ArmandLfd/nerfbaselines](https://github.com/ArmandLfd/nerfbaselines).

### 2.1 Create the Conda Environment
```bash
git submodule update --init --remote external/nerfbaselines
cd external/nerfbaselines
conda create -n nerfbaselines -y python=3.13
conda activate nerfbaselines
```

### 2.2 Install the API
```bash
pip install -e .
```

---

## 3. Installing Rendering Methods with Nerfbaselines


Install any supported method:

```bash
nerfbaselines install-method --method [gaussian-splatting|2d-gaussian-splatting|triangle-splatting|diff-dibr] --backend [conda|apptainer]
```

> [!NOTE]
We recommend installing methods inside an Apptainer container for reproducibility.
Our &#8706;DIBR method also works in a standard conda environment.

> [!WARNING]
If you installed `mamba` instead of `conda`, please do not use the `apptainer` backend, use rather the `conda` backend. Or install `conda`.

> [!WARNING]
If you want to use this code on a HPC, please try to remove some bindings from [nerfbaselines/backend/_apptainer.py](https://github.com/ArmandLfd/nerfbaselines/blob/main/nerfbaselines/backends/_apptainer.py).

Once completed, you can test the installation:

```bash
nerfbaselines test-method --method [gaussian-splatting|2d-gaussian-splatting|triangle-splatting|diff-dibr] --backend [conda|apptainer] --data external://llff/fern
```

On the first run, the tool downloads:
- Required datasets
- LPIPS pretrained weights

> [!WARNING] 
**Image pull permissions**: Some systems restrict image pulls to privileged users. See: [https://github.com/AndreWeiner/ml-cfd-lecture/issues/36](https://github.com/AndreWeiner/ml-cfd-lecture/issues/36).

> [!NOTE]
**Vulkan GPU selection**: Install `vulkan-tools` and run `vkinfo` to verify the active GPU.
If the system selects an integrated GPU, remove MESA Vulkan drivers and reinstall NVIDIA Vulkan drivers.

---

## 4. Dataset Download and Preparation

### 4.1 LLFF Dataset

Automatic download:

```bash
nerfbaselines download-dataset --data external://llff
```

The dataset will be download in the default cache folder, e.g., ```~/.cache/nerfbaselines/datasets/llff```.

### 4.2 Shiny Dataset

This dataset must be downloaded manually.

1. Download the Shiny dataset from [shiny_shared_folder](https://vistec-my.sharepoint.com/:f:/g/personal/pakkapon_p_s19_vistec_ac_th/EnIUhsRVJOdNsZ_4smdhye0B8z0VlxqOR35IR3bp0uGupQ?e=TsaQgM).
2. Once done, Copy each scene folder into ```~/.cache/nerfbaselines/datasets/shiny/{scene_name}```.
3. Run the script [resize.py](scripts/data/shiny/resize.py) to resize the images.

---

## 5. Training Ours and Baselines Methods

### 5.1 Default Training

To run a training/optimization with the default parameters of any method:

```bash
nerfbaselines train --method [gaussian-splatting|2d-gaussian-splatting|triangle-splatting|diff-dibr] --backend [conda|apptainer] --data ~/.cache/nerfbaselines/datasets/${DATASET_NAME}/${SCENE_NAME} --output ${OUTPUT}
```

> [!NOTE]
 **Vulkan Assert failure**: Our code can crash with this error: "vulkan Assert failure". Please first check that you have a correct installation of your GPU drivers and Vulkan installation. If the error persists, check if your GPU has enough memory (for full image resolution, at least 24GB of memory). If not, you can use `--set scale_factor=${FACTOR}` to rescale images and reduce GPU memory (e.g., `$FACTOR = 0.5`). 

### 5.2 Our Coarse-to-Fine Optimization

Use [our script](scripts/diff-dibr/coarse-to-fine.sh) implementing the multi‑stage optimization described in the paper.

To run the script:
```bash
chmod u +x scripts/diff-dibr/coarse-to-fine.sh
scripts/diff-dibr/coarse-to-fine.sh --data ~/.cache/nerfbaselines/datasets/${DATASET_NAME}/${SCENE_NAME} --output ${OUTPUT} --backend [conda|apptainer]
```

#### 5.2.1 Optional Parameters


| Parameters      | Description | Type | Required |
| ----------- | ----------- | ----------- | ----------- |
| --data      | Path to input data       | Path          | Yes |
| --output   | Output directory       | Path          | Yes |
| --backend   | Backend used to run the code        | [conda, apptainer]          | Yes |
| --num_iters   | Number of iterations in lvl 1 (lvl2 = 5*lvl1/3, lvl3 = 2*lvl1)        | numeric| No          |
| --depth_method   | Depth initialization        | [Zoe, DepthAnything3, NONE]          |  No |
| --nb_cams   | Nb of cams representing the scene        | numeric          |  No |
| --nb_a_cams   | Number of active cameras per rendering        | numeric          |  No |
| --l_atv_12   | Depth regularization weight (stages 1 & 2)        | numeric          |  No |
| --l_atv_3   | Depth regularization weight (stage 3)        | numeric          |  No |

---

## 7. Interactive Viewer

Launch the viewer:

```bash
nerfbaselines viewer --method [gaussian-splatting|2d-gaussian-splatting|triangle-splatting|diff-dibr] --backend [conda|apptainer] --data ~/.cache/nerfbaselines/datasets/${DATASET_NAME}/${SCENE_NAME} --output ${OUTPUT}/checkpoint-${NUM_ITERATIONS} --port 6064
```

Then, open [http://localhost:6064](http://localhost:6064) in your browser.

> [!NOTE]
By default, `${NUM_ITERATIONS}` equals `30000` for splatting-based methods. For ours, it equals `40000`. For the lower optimization stages, diminish the value to `20000` (lvl 1) or `30000` (lvl 2).

---

## 8. Results

The rendering results can be found in `${OUTPUT}/predictions-${NUM_ITERATIONS}`.

---

## 9. Standalone Installation Without NerfBaselines

While we recommend using NerfBaselines for reproducibility, standalone use is supported.

### 9.1 Installation

```bash
unzip -o ./diff_dibr.zip
cd diff_dibr
conda create -n diff-dibr -y python=3.13
conda activate diff-dibr

pip install -r requirements.txt

pip install \
    torch==2.9.1 \
    torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cu128

pip install lpips==0.1.4

pip install -e .
```

#### 9.1.1 Depth Submodules

Each depth module requires a dedicated conda environment named after the module:
- for ZoeDepth, the conda env must be named ```zoe```. 
- For DepthAnything3, it must be ```depthanything3```.

##### 9.1.1.1 ZoeDepth Installation

```bash
git submodule update --init --remote external/ZoeDepth
cd external/ZoeDepth
conda create -n zoe -y python=3.9.7
conda activate zoe
conda install -y \
    cuda=11.7.1 \
    h5py=3.7.0 \
    hdf5=1.12.2 \
    matplotlib=3.6.2 \
    matplotlib-base=3.6.2 \
    numpy=1.26.4 \
    scipy=1.10.0 \
    wandb=0.13.9 \
    tqdm=4.64.1 \
    huggingface_hub=0.11.1 \
    timm=0.6.12 \
    pip \
    -c pytorch \
    -c nvidia \
    -c conda-forge

pip install opencv-python==4.6.0.66

pip install \
    torch==1.13.1 \
    torchvision==0.14.1 \
    torchaudio==0.13.1 \
    --index-url https://download.pytorch.org/whl/cu117
```

##### 9.1.1.2 DepthAnything3 Installation

```bash
git submodule update --init --remote external/DepthAnything3
cd external/DepthAnything3
conda create -n depthanything3 -y python=3.12.12
conda activate depthanything3

conda install -y pip
pip install xformers torch\>=2 torchvision pillow \
  --index-url https://download.pytorch.org/whl/cu128

pip install -e .
```

### 9.2 Optimization
For optimization, please use NerfBaselines.

### 9.3 Standalone Viewer

```bash
python ./script/diff-dibr/freeview.py --path $OUTPUT/checkpoint-${NUM_ITERATIONS} --nb_cams_synthesis $NB_ACTIVE_CAMS
```

---

## 10. Uninstallation

### 10.1 Remove conda environment

```bash
conda remove --all -n nerfbaselines -y
```

### 10.2 Delete Nerfbaselines folder

```bash
sudo rm -r ~/.cache/nerfbaselines
```

### 10.3 Optional: Cleaning conda cache

```bash
conda clean --all -y
conda clean --index-cache
conda clean --packages
```

---

## Acknowledgments

Armand Losfeld is a FRIA grantee and Sarah Dury is a research fellow of the Fonds de la Recherche Scientifique – FNRS, Belgium. This work was supported by the 24 Top-Tier International Global Collaborative Research Program funded by ETRI; the IITP grant funded by the Korean government (MSIT) (RS-2017-II170072); and the Service Public de Wallonie Recherche – Wal4XR by Win4Excellence (2310144), Belgium.

We also extend our sincere gratitude to the developers and maintainers of the frameworks, methods, and datasets that made this work possible. Their efforts in building high‑quality, open‑source resources have been invaluable for advancing research in neural rendering and novel view synthesis.

- **RVS** - [https://gitlab.com/mpeg-i-visual/rvs](https://gitlab.com/mpeg-i-visual/rvs)
- **NerfBaselines** - [https://github.com/nerfbaselines/nerfbaselines](https://github.com/nerfbaselines/nerfbaselines)
- **3D Gaussian Splatting** — [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- **2D Gaussian Splatting** — [https://github.com/hbb1/2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting)
- **Triangle Splatting** — [https://github.com/trianglesplatting/triangle-splatting](https://github.com/trianglesplatting/triangle-splatting)
- **LLFF Dataset** — [https://github.com/Fyusion/LLFF](https://github.com/Fyusion/LLFF)
- **Shiny Dataset (NeX)** — [https://github.com/nex-mpi/nex-code](https://github.com/nex-mpi/nex-code)
- **Slang and Slangpy** - [https://github.com/shader-slang/slang](https://github.com/shader-slang/slang) and [https://github.com/shader-slang/slangpy](https://github.com/shader-slang/slangpy) 



---

## 12. License
Our project is under [**AGPL-3.0-or-later**](LICENSE). Please also read the [**copyright notice file**](NOTICE).

---

## 13. Dataset Licenses
- **LLFF** is under GPL-3.0 license 
- **Shiny (NeX)** is under MIT License
