#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "elu.hpp"
#include <algorithm>

#define LOCAL_SZ_X 1024
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct EluParam {
			int total;
			float alpha;
		};

		Elu::Elu(float _alpha) : m_alpha(_alpha) {
			layer::initVulkanThing(2);
			m_type = "Elu";
		}

		void Elu::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Elu::forward(std::vector<tensor>& ins, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool Elu::forward(tensor& in, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::elu_spv, sizeof(shaders::elu_spv));
				createPipeline(sizeof(EluParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, out, 1, m_descriptor_set);
			EluParam param = { m_total, m_alpha };
			recordCommandBuffer((void*)& param, sizeof(EluParam));
			runCommandBuffer();
			return true;
		}

		bool Elu::computeGroupCount() {
			m_group_x = (int)alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}