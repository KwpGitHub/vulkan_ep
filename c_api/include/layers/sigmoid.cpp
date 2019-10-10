#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "sigmoid.hpp"
#include <algorithm>

#define LOCAL_SZ_X 32
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct SigmoidParam {
			int total;
		};

		Sigmoid::Sigmoid() {
			layer::initVulkanThing(2);
			m_type = "Sigmoid";
		}

		void Sigmoid::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Sigmoid::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool Sigmoid::forward(tensor& in, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::sigmoid_spv, sizeof(shaders::sigmoid_spv));
				createPipeline(sizeof(SigmoidParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, out, 1, m_descriptor_set);
			SigmoidParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(SigmoidParam));
			runCommandBuffer();
			return true;
		}

		bool Sigmoid::computeGroupCount() {
			m_group_x = (int)alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}