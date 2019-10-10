#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "leakyrelu.hpp"
#include <algorithm>

#define LOCAL_SZ_X 32
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct LeakyReLUParams {
			int total;
			float slope;
		};

		LeakyReLU::LeakyReLU(float _alpha) : m_alpha(_alpha) {
			layer::initVulkanThing(2);
			m_type = "LeakyReLU";
		}

		void LeakyReLU::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool LeakyReLU::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool LeakyReLU::forward(tensor& in, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::leakyrelu_spv, sizeof(shaders::leakyrelu_spv));
				createPipeline(sizeof(LeakyReLUParams));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, out, 1, m_descriptor_set);
			LeakyReLUParams param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(LeakyReLUParams));
			runCommandBuffer();
			return true;
		}

		bool LeakyReLU::computeGroupCount() {
			m_group_x =(int)alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}