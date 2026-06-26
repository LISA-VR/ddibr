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

from abc import ABC, abstractmethod
from typing import Any

from ._utils import Logger

class LearningRate(ABC):

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> float:
        pass

    @abstractmethod
    def _update(self, *args: Any, **kwds: Any) -> float:
        pass

class CsteLearningRate(LearningRate):

    def __init__(self, lr : float):
        self.learning_rate = lr

    def __call__(self, *args: Any, **kwds: Any) -> float:
        return self._update()
    
    def _update(self, *args: Any, **kwds: Any) -> float:
        return self.learning_rate
    
class ExponentialLearningRate(LearningRate):

    def __init__(self, init_lr : float, decay_freq : float, decay_factor : float = 0.95):
        self.learning_rate = init_lr
        self.decay_freq = decay_freq
        self.decay_factor = decay_factor

    def __call__(self, step : float,*args: Any, **kwds: Any) -> float:
        return self._update(step)
    
    def _update(self, step : float, *args: Any, **kwds: Any) -> float:
        return self.learning_rate * self.decay_factor**(step / self.decay_freq)