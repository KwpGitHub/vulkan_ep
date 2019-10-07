#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "conv.hpp"
#include <algorithm>

#define LOCAL_SZ_X 32
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct AvgPooling1dParam {
			int channels;
            int in_w;
            int out_w;
            int padding_w;
            int filter_w;
            int stride_w;
            int total;
            int padded_area;
		};

		AvgPooling1d::AvgPooling1d() {
			layer::initVulkanThing(2);
			m_type = "AvgPooling1d";
		}

		void AvgPooling1d::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool AvgPooling1d::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool AvgPooling1d::forward(tensor& x, tensor& y) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::avgpool3d_spv, sizeof(shaders::avgpool3d_spv));
				createPipeline(sizeof(AvgPooling1dParam));
			}

			bindTensor(m_device, x, 0, m_descriptor_set);			
			bindTensor(m_device, y, 1, m_descriptor_set);

			AvgPooling1dParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(AvgPooling1dParam));
			runCommandBuffer();
			return true;
		}

		bool AvgPooling1d::computeGroupCount() {
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
		struct AvgPooling2dParam {
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
            int padded_area;
		};

		AvgPooling2d::AvgPooling2d() {
			layer::initVulkanThing(2);
			m_type = "AvgPooling2d";
		}

		void AvgPooling2d::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool AvgPooling2d::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool AvgPooling2d::forward(tensor& x,tensor& y) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::avgpool2d_spv, sizeof(shaders::avgpool2d_spv));
				createPipeline(sizeof(AvgPooling2dParam));
			}

			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, y, 1, m_descriptor_set);

			AvgPooling2dParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(AvgPooling2dParam));
			runCommandBuffer();
			return true;
		}

		bool AvgPooling2d::computeGroupCount() {
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
		struct AvgPooling3dParam {
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
            int padded_area;
		};

		AvgPooling3d::AvgPooling3d() {
			layer::initVulkanThing(2);
			m_type = "AvgPooling3d";
		}

		void AvgPooling3d::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool AvgPooling3d::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], ints[1], ins[3], outs[0]);
		}

		bool AvgPooling3d::forward(tensor& x, tensor& y) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::avgpool3d_spv, sizeof(shaders::avgpool3d_spv));
				createPipeline(sizeof(AvgPooling3dParam));
			}

			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, y, 1, m_descriptor_set);

			AvgPooling3dParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(AvgPooling3dParam));
			runCommandBuffer();
			return true;
		}

		bool AvgPooling3d::computeGroupCount() {
			m_group_x = alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}