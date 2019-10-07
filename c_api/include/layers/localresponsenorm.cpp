#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "localresponsenorm.hpp"
#include <algorithm>

#define LOCAL_SZ_X 32
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct LocalResponseNorm1dParam {
			int thread_num;
            int channels;
            int width;
            int filter_len;
            int radius;
            float alpha;
            float bias;
            float negative_beta;
		};

		LocalResponseNorm1d::LocalResponseNorm1d() {
			layer::initVulkanThing(2);
			m_type = "LocalResponseNorm1d";
		}

		void LocalResponseNorm1d::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool LocalResponseNorm1d::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool LocalResponseNorm1d::forward(tensor& x, tensor& y) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::localresponsenorm1d_spv, sizeof(shaders::localresponsenorm1d_spv));
				createPipeline(sizeof(LocalResponseNorm1dParam));
			}

			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, y, 2, m_descriptor_set);

			LocalResponseNorm1dParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(LocalResponseNorm1dParam));
			runCommandBuffer();
			return true;
		}

		bool LocalResponseNorm1d::computeGroupCount() {
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
		struct LocalResponseNorm2dParam {
			int thread_num;
            int channels;
            int height;
            int width;
            int filter_len;
            int radius;
            float alpha;
            float bias;
            float negative_beta;
		};

		LocalResponseNorm2d::LocalResponseNorm2d() {
			layer::initVulkanThing(4);
			m_type = "LocalResponseNorm2d";
		}

		void LocalResponseNorm2d::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool LocalResponseNorm2d::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool LocalResponseNorm2d::forward(tensor& x, tensor& y) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::localresponsenorm2d_spv, sizeof(shaders::localresponsenorm2d_spv));
				createPipeline(sizeof(LocalResponseNorm2dParam));
			}

			bindTensor(m_device, x, 0, m_descriptor_set);			
			bindTensor(m_device, y, 2, m_descriptor_set);

			LocalResponseNorm2dParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(LocalResponseNorm2dParam));
			runCommandBuffer();
			return true;
		}

		bool LocalResponseNorm2d::computeGroupCount() {
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
		struct LocalResponseNorm3dParam {
            int thread_num;
            int channels;
            int height;
            int width;
            int depth;
            int filter_len;
            int radius;
            float alpha;
            float bias;
            float negative_beta;
		};

		LocalResponseNorm3d::LocalResponseNorm3d() {
			layer::initVulkanThing(4);
			m_type = "LocalResponseNorm3d";
		}

		void LocalResponseNorm3d::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool LocalResponseNorm3d::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], ints[1], ins[3], outs[0]);
		}

		bool LocalResponseNorm3d::forward(tensor& x, tensor& y) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::localresponsenorm3d_spv, sizeof(shaders::localresponsenorm3d_spv));
				createPipeline(sizeof(LocalResponseNorm3dParam));
			}

			bindTensor(m_device, x, 0, m_descriptor_set);
			bindTensor(m_device, y, 1, m_descriptor_set);

			LocalResponseNorm3dParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(LocalResponseNorm3dParam));
			runCommandBuffer();
			return true;
		}

		bool LocalResponseNorm3d::computeGroupCount() {
			m_group_x = alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}