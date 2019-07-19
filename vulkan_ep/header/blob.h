// Tencent is pleased to support the open source community by making vulkan_ep available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef VK_EP_BLOB_H
#define VK_EP_BLOB_H

#include <string>
#include <vector>
#include "platform.h"

namespace vulkan_ep {

class Blob
{
public:
    // empty
    Blob();

public:
#if VK_EP_STRING
    // blob name
    std::string name;
#endif // VK_EP_STRING
    // layer index which produce this blob as output
    int producer;
    // layer index which need this blob as input
    std::vector<int> consumers;
};

} // namespace vulkan_ep

#endif // VK_EP_BLOB_H
