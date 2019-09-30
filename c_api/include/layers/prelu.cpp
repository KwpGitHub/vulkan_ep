#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "prelu.hpp"
#include <algorithm>

#define LOCAL_SZ_X 32
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct PReLUParam {
			int total;
			float slope;
		};

		PReLU::PReLU(float _slope) : m_slope(_slope) {
			layer::initVulkanThing(2);
			m_type = "PReLU";
		}

		void PReLU::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool PReLU::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool PReLU::forward(tensor& in, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::prelu_spv, sizeof(shaders::prelu_spv));
				createPipeline(sizeof(PReLUParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, out, 1, m_descriptor_set);
			PReLUParam param = { m_total, m_slope };
			recordCommandBuffer((void*)& param, sizeof(PReLUParam));
			runCommandBuffer();
			return true;
		}

		bool PReLU::computeGroupCount() {
			m_group_x = alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}