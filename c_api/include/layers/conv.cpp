#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "conv.hpp"
#include <algorithm>

#define LOCAL_SZ_X 32
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct Conv1dParam {
			int in_w;
            int out_w;
            int stride_w;
            int pad_w;
            int filter_w;
            int dilation_w;
            int channels;
            int batch;
            int has_bias;
            int K; //filter_width * in_channel
            int N; //out_channel
            int basic_shader_batch_idx;
            int basic_shader_partition_idx;
            int basic_shader_partition_size;
		};

		Conv1d::Conv1d() {
			layer::initVulkanThing(4);
			m_type = "Conv1d";
		}

		void Conv1d::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Conv1d::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool Conv1d::forward(tensor& x, tensor& w, tensor& b, tensor& y) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::conv1d_spv, sizeof(shaders::conv1d_spv));
				createPipeline(sizeof(Conv1dParam));
			}

			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, w, 1, m_descriptor_set);
            bindTensor(m_device, b, 2, m_descriptor_set);
			bindTensor(m_device, y, 3, m_descriptor_set);

			Conv1dParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(Conv1dParam));
			runCommandBuffer();
			return true;
		}

		bool Conv1d::computeGroupCount() {
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
		struct Conv2dParam {
			int in_h;
            int in_w;
            int out_h;
            int out_w;
            int stride_h;
            int stride_w;
            int pad_h;
            int pad_w;
            int filter_h;
            int filter_w;
            int dilation_h;
            int dilation_w;
            int channels;
            int batch;
            int has_bias;
            int M; //out_h * out_w
            int K; //filter_h * filter_w * in_channel
            int N; //out_channel
            int basic_shader_batch_idx;
            int basic_shader_partition_idx;
            int basic_shader_partition_size;
		};

		Conv2d::Conv2d() {
			layer::initVulkanThing(4);
			m_type = "Conv2d";
		}

		void Conv2d::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Conv2d::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool Conv2d::forward(tensor& x, tensor& w, tensor& b, tensor& y) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::conv2d_spv, sizeof(shaders::conv2d_spv));
				createPipeline(sizeof(Conv2dParam));
			}

			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, w, 1, m_descriptor_set);
            bindTensor(m_device, b, 2, m_descriptor_set);
			bindTensor(m_device, y, 3, m_descriptor_set);

			Conv2dParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(Conv2dParam));
			runCommandBuffer();
			return true;
		}

		bool Conv2d::computeGroupCount() {
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
		struct Conv3dParam {
			int in_h;
            int in_w;
            int in_d;
            int out_h;
            int out_w;
            int out_d;
            int stride_h;
            int stride_w;
            int stride_d;
            int pad_h;
            int pad_w;
            int pad_d;
            int filter_h;
            int filter_w;
            int filter_d;
            int dilation_h;
            int dilation_w;
            int dilation_d;
            int channels;
            int batch;
            int has_bias;
            int M; //out_h * out_w * out_d;
            int K; //filter_h * filter_w * filter_d * in_channel
            int N; //out_channel
            int basic_shader_batch_idx;
            int basic_shader_partition_idx;
            int basic_shader_partition_size;
		};

		Conv3d::Conv3d() {
			layer::initVulkanThing(4);
			m_type = "Conv3d";
		}

		void Conv3d::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Conv3d::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], ints[1], ins[3], outs[0]);
		}

		bool Conv3d::forward(tensor& x, tensor& w, tensor& b, tensor& y) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::conv3d_spv, sizeof(shaders::conv3d_spv));
				createPipeline(sizeof(Conv3dParam));
			}

			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, w, 1, m_descriptor_set);
            bindTensor(m_device, b, 2, m_descriptor_set);
			bindTensor(m_device, y, 3, m_descriptor_set);

			Conv3dParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(Conv3dParam));
			runCommandBuffer();
			return true;
		}

		bool Conv3d::computeGroupCount() {
			m_group_x = alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}