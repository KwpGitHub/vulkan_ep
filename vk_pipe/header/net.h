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

#ifndef VK_EP_NET_H
#define VK_EP_NET_H

#include <stdio.h>
#include <vector>
#include "platform.h"
#include "blob.h"
#include "layer.h"
#include "mat.h"
#include "option.h"

namespace vulkan_ep {

#if VK_EP_VULKAN
class VkCompute;
#endif // VK_EP_VULKAN
class Extractor;
class Net
{
public:
    // empty init
    Net();
    // clear and destroy
    ~Net();

public:
    // option can be changed before loading
    Option opt;

#if VK_EP_VULKAN
    // set gpu device by index
    void set_vulkan_device(int device_index);

    // set gpu device by device handle, no owner transfer
    void set_vulkan_device(const VulkanDevice* vkdev);
#endif // VK_EP_VULKAN

#if VK_EP_STRING
    // register custom layer by layer type name
    // return 0 if success
    int register_custom_layer(const char* type, layer_creator_func creator);
#endif // VK_EP_STRING
    // register custom layer by layer type
    // return 0 if success
    int register_custom_layer(int index, layer_creator_func creator);

#if VK_EP_STDIO
#if VK_EP_STRING
    // load network structure from plain param file
    // return 0 if success
    int load_param(FILE* fp);
    int load_param(const char* protopath);
    int load_param_mem(const char* mem);
#endif // VK_EP_STRING
    // load network structure from binary param file
    // return 0 if success
    int load_param_bin(FILE* fp);
    int load_param_bin(const char* protopath);

    // load network weight data from model file
    // return 0 if success
    int load_model(FILE* fp);
    int load_model(const char* modelpath);
#endif // VK_EP_STDIO

    // load network structure from external memory
    // memory pointer must be 32-bit aligned
    // return bytes consumed
    int load_param(const unsigned char* mem);

    // reference network weight data from external memory
    // weight data is not copied but referenced
    // so external memory should be retained when used
    // memory pointer must be 32-bit aligned
    // return bytes consumed
    int load_model(const unsigned char* mem);

    // unload network structure and weight data
    void clear();

    // construct an Extractor from network
    Extractor create_extractor() const;

protected:
    // parse the structure of network
    // fuse int8 op dequantize and quantize by requantize
    int fuse_network();

#if VK_EP_VULKAN

    int upload_model();

    int create_pipeline();

    int destroy_pipeline();

#endif // VK_EP_VULKAN

    friend class Extractor;
#if VK_EP_STRING
    int find_blob_index_by_name(const char* name) const;
    int find_layer_index_by_name(const char* name) const;
    int custom_layer_to_index(const char* type);
    Layer* create_custom_layer(const char* type);
#endif // VK_EP_STRING
    Layer* create_custom_layer(int index);
    int forward_layer(int layer_index, std::vector<Mat>& blob_mats, Option& opt) const;

#if VK_EP_VULKAN
    int forward_layer(int layer_index, std::vector<Mat>& blob_mats, std::vector<VkMat>& blob_mats_gpu, VkCompute& cmd, Option& opt) const;
#endif // VK_EP_VULKAN

protected:
    std::vector<Blob> blobs;
    std::vector<Layer*> layers;

    std::vector<layer_registry_entry> custom_layer_registry;

#if VK_EP_VULKAN
    const VulkanDevice* vkdev;

    VkAllocator* weight_vkallocator;
    VkAllocator* weight_staging_vkallocator;

    vulkan_ep::Layer* cast_float32_to_float16;
    vulkan_ep::Layer* cast_float16_to_float32;
    vulkan_ep::Layer* packing_pack1;
    vulkan_ep::Layer* packing_pack4;
#endif // VK_EP_VULKAN
};

class Extractor
{
public:
    // enable light mode
    // intermediate blob will be recycled when enabled
    // enabled by default
    void set_light_mode(bool enable);

    // set thread count for this extractor
    // this will overwrite the global setting
    // default count is system depended
    void set_num_threads(int num_threads);

    // set blob memory allocator
    void set_blob_allocator(Allocator* allocator);

    // set workspace memory allocator
    void set_workspace_allocator(Allocator* allocator);

#if VK_EP_VULKAN
    void set_vulkan_compute(bool enable);

    void set_blob_vkallocator(VkAllocator* allocator);

    void set_workspace_vkallocator(VkAllocator* allocator);

    void set_staging_vkallocator(VkAllocator* allocator);
#endif // VK_EP_VULKAN

#if VK_EP_STRING
    // set input by blob name
    // return 0 if success
    int input(const char* blob_name, const Mat& in);

    // get result by blob name
    // return 0 if success
    int extract(const char* blob_name, Mat& feat);
#endif // VK_EP_STRING

    // set input by blob index
    // return 0 if success
    int input(int blob_index, const Mat& in);

    // get result by blob index
    // return 0 if success
    int extract(int blob_index, Mat& feat);

#if VK_EP_VULKAN
#if VK_EP_STRING
    // set input by blob name
    // return 0 if success
    int input(const char* blob_name, const VkMat& in);

    // get result by blob name
    // return 0 if success
    int extract(const char* blob_name, VkMat& feat, VkCompute& cmd);
#endif // VK_EP_STRING

    // set input by blob index
    // return 0 if success
    int input(int blob_index, const VkMat& in);

    // get result by blob index
    // return 0 if success
    int extract(int blob_index, VkMat& feat, VkCompute& cmd);
#endif // VK_EP_VULKAN

protected:
    friend Extractor Net::create_extractor() const;
    Extractor(const Net* net, int blob_count);

private:
    const Net* net;
    std::vector<Mat> blob_mats;
    Option opt;

#if VK_EP_VULKAN
    std::vector<VkMat> blob_mats_gpu;
#endif // VK_EP_VULKAN
};

} // namespace vulkan_ep

#endif // VK_EP_NET_H
