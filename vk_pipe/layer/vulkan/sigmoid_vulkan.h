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

#ifndef LAYER_SIGMOID_VULKAN_H
#define LAYER_SIGMOID_VULKAN_H

#include "sigmoid.h"

namespace vulkan_ep {

class Sigmoid_vulkan : virtual public Sigmoid
{
public:
    Sigmoid_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_sigmoid;
    Pipeline* pipeline_sigmoid_pack4;
};

} // namespace vulkan_ep

#endif // LAYER_SIGMOID_VULKAN_H
