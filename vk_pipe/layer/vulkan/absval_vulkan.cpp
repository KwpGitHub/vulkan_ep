// Tencent is pleased to support the open source community by making vulkan_ep available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "absval_vulkan.h"

namespace vulkan_ep {

DEFINE_LAYER_CREATOR(AbsVal_vulkan)

AbsVal_vulkan::AbsVal_vulkan()
{
    support_vulkan = true;

    pipeline_absval = 0;
    pipeline_absval_pack4 = 0;
}

int AbsVal_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> specializations;

    // pack1
    {
        pipeline_absval = new Pipeline(vkdev);
        pipeline_absval->set_optimal_local_size_xyz();
        pipeline_absval->create("absval", opt, specializations, 1, 5);
    }

    // pack4
    {
        pipeline_absval_pack4 = new Pipeline(vkdev);
        pipeline_absval_pack4->set_optimal_local_size_xyz();
        pipeline_absval_pack4->create("absval_pack4", opt, specializations, 1, 5);
    }

    return 0;
}

int AbsVal_vulkan::destroy_pipeline(const Option& opt)
{
    delete pipeline_absval;
    pipeline_absval = 0;

    delete pipeline_absval_pack4;
    pipeline_absval_pack4 = 0;

    return 0;
}

int AbsVal_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    int packing = bottom_top_blob.packing;

    std::vector<VkMat> bindings(1);
    bindings[0] = bottom_top_blob;

    std::vector<vk_constant_type> constants(5);
    constants[0].i = bottom_top_blob.dims;
    constants[1].i = bottom_top_blob.w;
    constants[2].i = bottom_top_blob.h;
    constants[3].i = bottom_top_blob.c;
    constants[4].i = bottom_top_blob.cstep;

    const Pipeline* pipeline = packing == 4 ? pipeline_absval_pack4 : pipeline_absval;

    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);

    return 0;
}

} // namespace vulkan_ep
