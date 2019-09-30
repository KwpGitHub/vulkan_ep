#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "exp.hpp"
#include <algorithm>

#define LOCAL_SZ_X 32
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct ExpParam {
			int total;
		};

		Exp::Exp() {
			layer::initVulkanThing(2);
			m_type = "Exp";
		}

		void Exp::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Exp::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool Exp::forward(tensor& in, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::exp_spv, sizeof(shaders::exp_spv));
				createPipeline(sizeof(ExpParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, out, 1, m_descriptor_set);
			ExpParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(ExpParam));
			runCommandBuffer();
			return true;
		}

		bool Exp::computeGroupCount() {
			m_group_x = alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}