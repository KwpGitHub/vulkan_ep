#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "atan.hpp"
#include <algorithm>

#define LOCAL_SZ_X 1024
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct AtanParam {
			int total;
		};

		Atan::Atan() {
			layer::initVulkanThing(2);
			m_type = "Atan";
		}

		void Atan::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Atan::forward(std::vector<tensor>& ins, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool Atan::forward(tensor& in, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::atan_spv, sizeof(shaders::atan_spv));
				createPipeline(sizeof(AtanParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, out, 1, m_descriptor_set);
			AtanParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(AtanParam));
			runCommandBuffer();
			return true;
		}

		bool Atan::computeGroupCount() {
			m_group_x = (int)alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}