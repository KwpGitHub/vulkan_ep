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

#include "binaryop_vulkan.h"
#include <math.h>
#include <algorithm>
#include <functional>

namespace vulkan_ep {

DEFINE_LAYER_CREATOR(BinaryOp_vulkan)

BinaryOp_vulkan::BinaryOp_vulkan()
{
    support_vulkan = true;

    pipeline_binaryop = 0;
    pipeline_binaryop_pack4 = 0;
}

int BinaryOp_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> specializations(3);
    specializations[0].i = op_type;
    specializations[1].i = with_scalar;
    specializations[2].f = b;

    // pack1
    {
        pipeline_binaryop = new Pipeline(vkdev);
        pipeline_binaryop->set_optimal_local_size_xyz();
        pipeline_binaryop->create("binaryop", opt, specializations, 3, 15);
    }

    // pack4
    {
        pipeline_binaryop_pack4 = new Pipeline(vkdev);
        pipeline_binaryop_pack4->set_optimal_local_size_xyz();
        pipeline_binaryop_pack4->create("binaryop_pack4", opt, specializations, 3, 15);
    }

    return 0;
}

int BinaryOp_vulkan::destroy_pipeline(const Option& opt)
{
    delete pipeline_binaryop;
    pipeline_binaryop = 0;

    delete pipeline_binaryop_pack4;
    pipeline_binaryop_pack4 = 0;

    return 0;
}

int BinaryOp_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& bottom_blob = bottom_blobs[0];
    const VkMat& bottom_blob1 = bottom_blobs[1];

    VkMat& top_blob = top_blobs[0];

    int packing = bottom_blob.packing;

    // TODO broadcast
    top_blob.create_like(bottom_blob, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(3);
    bindings[0] = bottom_blob;
    bindings[1] = bottom_blob1;
    bindings[2] = top_blob;

    std::vector<vk_constant_type> constants(15);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.c;
    constants[4].i = bottom_blob.cstep;
    constants[5].i = bottom_blob1.dims;
    constants[6].i = bottom_blob1.w;
    constants[7].i = bottom_blob1.h;
    constants[8].i = bottom_blob1.c;
    constants[9].i = bottom_blob1.cstep;
    constants[10].i = top_blob.dims;
    constants[11].i = top_blob.w;
    constants[12].i = top_blob.h;
    constants[13].i = top_blob.c;
    constants[14].i = top_blob.cstep;

    const Pipeline* pipeline = packing == 4 ? pipeline_binaryop_pack4 : pipeline_binaryop;

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

int BinaryOp_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    int packing = bottom_top_blob.packing;

    std::vector<VkMat> bindings(3);
    bindings[0] = bottom_top_blob;
    bindings[1] = bottom_top_blob;// TODO use dummy buffer
    bindings[2] = bottom_top_blob;// TODO use dummy buffer

    std::vector<vk_constant_type> constants(15);
    constants[10].i = bottom_top_blob.dims;
    constants[11].i = bottom_top_blob.w;
    constants[12].i = bottom_top_blob.h;
    constants[13].i = bottom_top_blob.c;
    constants[14].i = bottom_top_blob.cstep;

    const Pipeline* pipeline = packing == 4 ? pipeline_binaryop_pack4 : pipeline_binaryop;

    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);

    return 0;
}

} // namespace vulkan_ep
