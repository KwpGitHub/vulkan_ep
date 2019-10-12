#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "neg.hpp"
#include <algorithm>

#define LOCAL_SZ_X 1024
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct NegParam {
			int total;
		};

		Neg::Neg() {
			layer::initVulkanThing(2);
			m_type = "Neg";
		}

		void Neg::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Neg::forward(std::vector<tensor>& ins, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool Neg::forward(tensor& in, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::neg_spv, sizeof(shaders::neg_spv));
				createPipeline(sizeof(NegParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, out, 1, m_descriptor_set);
			NegParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(NegParam));
			runCommandBuffer();
			return true;
		}

		bool Neg::computeGroupCount() {
			m_group_x =(int)alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}