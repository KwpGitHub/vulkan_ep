#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "acosh.hpp"
#include <algorithm>

#define LOCAL_SZ_X 32
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct AcoshParam {
			int total;
		};

		Acosh::Acosh() {
			layer::initVulkanThing(2);
			m_type = "Acosh";
		}

		void Acosh::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Acosh::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool Acosh::forward(tensor& in, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::acosh_spv, sizeof(shaders::acosh_spv));
				createPipeline(sizeof(AcoshParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, out, 1, m_descriptor_set);
			AcoshParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(AcoshParam));
			runCommandBuffer();
			return true;
		}

		bool Acosh::computeGroupCount() {
			m_group_x = alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}