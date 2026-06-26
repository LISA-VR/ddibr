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

struct cst_type_value<T : __BuiltinFloatingPointType> {
    static constexpr T MAX_FLOAT32_VALUE() {
        constexpr T base = T(2);
        constexpr T exp_1 = T(-23);
        constexpr T exp_2 = T(-23);
        constexpr T value = (base - pow<T>(base, exp_1)) * pow<T>(base, exp_2);
        return T(value);
    }

    static constexpr T MAX_SINT32_VALUE() {
        constexpr T base = T(2);
        constexpr T exp = T(30);
        constexpr T value = pow<T>(base, exp) - T(1);
        return T(value);
    }

    static constexpr float EPS() { return 1e-8f;};
    static constexpr float HIGH_EPS() { return 1e-5f;};
};