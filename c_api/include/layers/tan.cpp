#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "tan.hpp"
#include <algorithm>

#define LOCAL_SZ_X 1024
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct TanParam {
			int total;
		};

		Tan::Tan() {
			layer::initVulkanThing(2);
			m_type = "Tan";
		}

		void Tan::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Tan::forward(std::vector<tensor>& ins, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool Tan::forward(tensor& in, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::tan_spv, sizeof(shaders::tan_spv));
				createPipeline(sizeof(TanParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, out, 1, m_descriptor_set);
			TanParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(TanParam));
			runCommandBuffer();
			return true;
		}

		bool Tan::computeGroupCount() {
			m_group_x =(int)alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}