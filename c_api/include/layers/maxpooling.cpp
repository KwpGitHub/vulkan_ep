#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "maxpooling.hpp"
#include <algorithm>

#define LOCAL_SZ_X 32
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct MaxPooling1dParam {
			int channels;      
            int in_w;
            int out_w;
            int padding_w;
            int filter_w;
            int stride_w;
            int total;
            int need_mask;
        } p;
		};

		MaxPooling1d::MaxPooling1d() {
			layer::initVulkanThing(3);
			m_type = "MaxPooling1d";
		}

		void MaxPooling1d::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool MaxPooling1d::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0], outs[0]);
		}

		bool MaxPooling1d::forward(tensor& x, tensor& y tensor& mask) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::maxpool1d_spv, sizeof(shaders::maxpool1d_spv));
				createPipeline(sizeof(MaxPooling3dParam));
			}

			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, y, 1, m_descriptor_set);
			bindTensor(m_device, mask, 2, m_descriptor_set);

			MaxPooling3dParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(MaxPooling3dParam));
			runCommandBuffer();
			return true;
		}

		bool MaxPooling1d::computeGroupCount() {
			m_group_x = alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}


namespace kernel {
	namespace layers {
		struct MaxPooling2dParam {
            int channels;
            int in_h;
            int in_w;
            int out_h;
            int out_w;
            int padding_h;
            int padding_w;
            int filter_h;
            int filter_w;
            int stride_h;
            int stride_w;
            int total;
            int need_mask;
		};

		MaxPooling2d::MaxPooling2d() {
			layer::initVulkanThing(3);
			m_type = "MaxPooling2d";
		}

		void MaxPooling2d::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool MaxPooling2d::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0], outs[1]);
		}

		bool MaxPooling2d::forward(tensor& x, tensor& y tensor& mask) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::maxpool2d_spv, sizeof(shaders::maxpool2d_spv));
				createPipeline(sizeof(MaxPooling3dParam));
			}

			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, y, 1, m_descriptor_set);
			bindTensor(m_device, mask, 2, m_descriptor_set);

			MaxPooling3dParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(MaxPooling3dParam));
			runCommandBuffer();
			return true;
		}

		bool MaxPooling2d::computeGroupCount() {
			m_group_x = alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}


namespace kernel {
	namespace layers {
		struct MaxPooling3dParam {
            int channels;
            int in_h;
            int in_w;
            int in_d;
            int out_h;
            int out_w;
            int out_d;
            int padding_h;
            int padding_w;
            int padding_d;
            int filter_h;
            int filter_w;
            int filter_d;
            int stride_h;
            int stride_w;
            int stride_d;
            int total;
            int need_mask;
		};

		MaxPooling3d::MaxPooling3d() {
			layer::initVulkanThing(3);
			m_type = "MaxPooling3d";
		}

		void MaxPooling3d::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool MaxPooling3d::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0], outs[1]);
		}

		bool MaxPooling3d::forward(tensor& x, tensor& y tensor& mask) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::maxpool3d_spv, sizeof(shaders::maxpool3d_spv));
				createPipeline(sizeof(MaxPooling3dParam));
			}

			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, y, 1, m_descriptor_set);
			bindTensor(m_device, mask, 2, m_descriptor_set);

			MaxPooling3dParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(MaxPooling3dParam));
			runCommandBuffer();
			return true;
		}

		bool MaxPooling3d::computeGroupCount() {
			m_group_x = alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}