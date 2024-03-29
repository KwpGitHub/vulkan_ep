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

#ifndef LAYER_ELTWISE_VULKAN_H
#define LAYER_ELTWISE_VULKAN_H

#include "eltwise.h"

namespace vulkan_ep {

class Eltwise_vulkan : virtual public Eltwise
{
public:
    Eltwise_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_eltwise[2];
    Pipeline* pipeline_eltwise_pack4[2];
};

} // namespace vulkan_ep

#endif // LAYER_ELTWISE_VULKAN_H
