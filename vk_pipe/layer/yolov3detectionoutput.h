// Tencent is pleased to support the open source community by making vulkan_ep available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef LAYER_YOLOV3DETECTIONOUTPUT_H
#define LAYER_YOLOV3DETECTIONOUTPUT_H

#include "layer.h"

namespace vulkan_ep {

class Yolov3DetectionOutput : public Layer
{
public:
	Yolov3DetectionOutput();
    ~Yolov3DetectionOutput();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    int num_class;
    int num_box;
    float confidence_threshold;
    float nms_threshold;
    Mat biases;
	Mat mask;
	Mat anchors_scale;
	int mask_group_num;
    vulkan_ep::Layer* softmax;
};

} // namespace vulkan_ep

#endif // LAYER_YOLODETECTIONOUTPUT_H
