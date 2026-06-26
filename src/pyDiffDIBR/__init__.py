"""
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

# Marks this directory as a Python package
from ._utils import (
    Logger
)

from .Image import (
    Image,
    RWImage,
    MultiSampledRenderedImage,
    RenderedImage,
    RenderedMultiSampleImageComposite,

    ImagePair,
    RWImagePair,
    MultiSampledRenderedImagePair,
    RenderedImagePair,
    RenderedMultiSampleImagePairComposite,
) 
from .Camera import (
    CameraParamDict,
    ProjectionType,
    CPUCameraParameters,
    GPUCameraParameters,
    Camera,
    RenderedViewpoint,
    BlendedViewpoint
) 
from .Dataset import (
    Dataset,
    DatasetIO,
    DatasetSplitUtils,
    DatasetSpatialUtils,
    DatasetTrainingUtils,
    DatasetCameraSelectorUtils,
    DatasetProcessingUtils,
    ColorDataset
)
from .RenderingApp import (
    RenderingObject,
    RenderingStageType,
    RenderingObjectType,
    RenderingApp,
)
from .RenderingDevice import (
    RenderingDevice
)
from .Window import (
    Window,
    FreeViewpointNavigatorWindow
)
from .Optimizer import (
    Optimizer,
    GradientDescent,
    Adam
)
from .LearningRate import (
    LearningRate,
    CsteLearningRate,
    ExponentialLearningRate
)
from .Loss import (
    Loss,
    L1Norm,
    L2Norm,
    MaskedL1Norm,
    AnisotropicTotalVariation,
    DepthConsistency
)
from .FwdRendering  import FwdRenderingApp
from .BwdRendering  import BwdRenderingApp
from .Model         import DiffDIBR

from .Trainer import (
    SimpleTrainer,
    RandomInputTrainingSampling,
    RandomInputTrainingSamplingMethod
)

from ._constants import (
    DEFAULT_TOML_DIR
)