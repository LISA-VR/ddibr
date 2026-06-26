#!/usr/bin/env bash

# This file is part of "∂DIBR", a differentiable renderer based on Depth Image Based Rendering (DIBR) techniques for fast Novel View Synthesis.
# Copyright (C) 2026 "Université Libre de Bruxelles (ULB)". All rights reserved.
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# 
# Contact:
#     - Armand Losfeld - armand.lfd.pro@proton.me
#     - Daniele Bonatto - daniele.bonatto@ulb.be


coarse_to_fine () {
    # $1 is data
    # $2 is output dir
    # $3 is num_iterations - 1
    # $4 is depth_method
    # $5 is nb_cams
    # $6 is nb_a_c
    # $7 is l_atv_12
    # $8 is l_atv_3
    # $9 is l_geo
    
    in_data=$1
    out_data=$2
    
    num_iters_1=$(($3))
    num_iters_2=$(($3 + 2 * $3 / 3))
    num_iters_3=$(($num_iters_2 + $3 / 3))
    
    depth_m=$4
    
    ckpt_2="${out_data}/checkpoint-${num_iters_1}"
    ckpt_3="${out_data}/checkpoint-${num_iters_2}"
    
    nb_cams=$5
    nb_a_cams_1=$6
    nb_a_cams_2=$6
    nb_a_cams_3=$6
    
    l_atv_1=$7
    l_atv_2=$7
    l_atv_3=$8
    
    l_loss_d_consistency=0
    
    lr_d_1=1e-2
    lr_d_2=1e-2
    lr_d_3=5e-3
    
    lr_c_1=1e-5
    lr_c_2=5e-5
    lr_c_3=5e-3
    
    scl_1=0.25
    scl_2=0.5
    scl_3=1.0
    
    scl_ckpt_1=1.0
    scl_ckpt_2=2.0
    scl_ckpt_3=2.0
    
    rnd_sampling_1=3
    rnd_sampling_2=3
    rnd_sampling_3=1
    
    pdd_1=15
    pdd_2=10
    pdd_3=10
    
    toml="default_front_facing.toml"

    backend=$9
    
    nerfbaselines train --method diff-dibr --backend $backend --eval-few-iters 1000::1000 --set reset_depth=1 --set l_loss_l2_color=0.0  --set toml_name=$toml --set l_loss_l1_color=1.0 --set l_loss_d_consistency=$l_loss_d_consistency \
        --data $in_data --output $out_data --set depth_method=$depth_m --set scale_factor=$scl_1 --set scale_factor_ckpt=$scl_ckpt_1  --set padding_input=$pdd_1 \
        --set num_iterations=$num_iters_1 --set nb_cams=$nb_cams --set nb_active_cams=$nb_a_cams_1 --set rnd_training_sampling=$rnd_sampling_1 \
        --set l_loss_d_atv=$l_atv_1  --set lr_depth=$lr_d_1 --set lr_color=$lr_c_1
    
    nerfbaselines train --method diff-dibr --backend $backend --eval-few-iters 1000::1000 --set l_loss_l2_color=0.0  --set toml_name=$toml --set l_loss_l1_color=1.0 --set l_loss_d_consistency=$l_loss_d_consistency \
        --data $in_data --output $out_data --set depth_method=$depth_m --set scale_factor=$scl_2 --set scale_factor_ckpt=$scl_ckpt_2  --set padding_input=$pdd_2 \
        --set num_iterations=$num_iters_2 --set nb_cams=$nb_cams --set nb_active_cams=$nb_a_cams_2 --set rnd_training_sampling=$rnd_sampling_2 \
        --set l_loss_d_atv=$l_atv_2  --set lr_depth=$lr_d_2 --set lr_color=$lr_c_2 --checkpoint $ckpt_2
        
    nerfbaselines train --method diff-dibr --backend $backend --eval-few-iters 1000::1000 --set l_loss_l2_color=0.0  --set toml_name=$toml --set l_loss_l1_color=1.0 --set l_loss_d_consistency=$l_loss_d_consistency \
        --data $in_data --output $out_data --set depth_method=$depth_m --set scale_factor=$scl_3 --set scale_factor_ckpt=$scl_ckpt_3  --set padding_input=$pdd_3 \
        --set num_iterations=$num_iters_3 --set nb_cams=$nb_cams --set nb_active_cams=$nb_a_cams_3 --set rnd_training_sampling=$rnd_sampling_3 \
        --set l_loss_d_atv=$l_atv_3  --set lr_depth=$lr_d_3 --set lr_color=$lr_c_3 --checkpoint $ckpt_3
}

# -------------------------------------------------------------------
# Generic argument parser for --key value pairs
# Usage: parse_args args_spec args_array
#   args_spec : associative array with argument names as keys and
#               specification strings as values:
#                 "required"           -> argument must be provided
#                 "default:<value>"     -> optional, use default if missing
#   args_array: the array of command line arguments ("$@")
# -------------------------------------------------------------------
parse_args () {
    # Declare local associative array reference (requires bash 4.3+ for nameref)
    local -n spec="$1"
    shift
    local -a args=("$@")

    # This associative array will hold the parsed values (optional, for verification)
    declare -A parsed

    # Loop over the arguments
    local i=0
    while [[ $i -lt ${#args[@]} ]]; do
        arg="${args[$i]}"

        # Check if it starts with --
        if [[ "$arg" == --* ]]; then
            # Remove leading --
            key="${arg#--}"

            # Ensure there is a following value
            if [[ $((i+1)) -ge ${#args[@]} ]] || [[ '${args[$((i+1))}' == --* ]]; then
                echo "Error: Missing value for argument '$arg'" >&2
                exit 1
            fi

            value="${args[$((i+1))]}"
            # Store in a variable named after the key
            printf -v "$key" "%s" "$value"
            # Also record in parsed array for validation
            parsed["$key"]="$value"

            # Skip the next argument (the value)
            i=$((i+2))
        else
            echo "Error: Unexpected argument '$arg' (all arguments must start with --)" >&2
            exit 1
        fi
    done

    # Now check required arguments and apply defaults for missing optional ones
    for key in "${!spec[@]}"; do
        spec_val="${spec[$key]}"
        if [[ -z "${parsed[$key]+x}" ]]; then
            # Argument not provided
            if [[ "$spec_val" == "required" ]]; then
                echo "Error: Required argument '--$key' is missing." >&2
                exit 1
            elif [[ "$spec_val" =~ ^default:(.*)$ ]]; then
                # Set default value
                default_val="${BASH_REMATCH[1]}"
                printf -v "$key" "%s" "$default_val"
                echo "Info: Argument '--$key' not provided, using default: $default_val" >&2
            else
                # No specification? Should not happen, but ignore.
                :
            fi
        fi
    done
}

# Define the specification for arguments
declare -A arg_spec=(
    [data]="required"
    [output]="required"
    [backend]="required"
    [num_iters]="default:20000"
    [depth_method]="default:Zoe"
    [nb_cams]="default:15"
    [nb_a_cams]="default:5"
    [l_atv_12]="default:4"
    [l_atv_3]="default:12"
)

# Parse the command line arguments (pass "$@")
parse_args arg_spec "$@"

coarse_to_fine $data $output $num_iters $depth_method $nb_cams $nb_a_cams $l_atv_12 $l_atv_3 $backend

# $1 is data
# $2 is output dir
# $3 is num_iterations - 1
# $4 is depth_method
# $5 is nb_cams
# $6 is nb_a_c
# $7 is l_atv_12
# $8 is l_atv_3
# $9 is backend
