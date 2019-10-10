#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "softsign.hpp"
#include <algorithm>

#define LOCAL_SZ_X 32
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct SoftsignParam {
			int total;
		};

		Softsign::Softsign() {
			layer::initVulkanThing(2);
			m_type = "Softsign";
		}

		void Softsign::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Softsign::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool Softsign::forward(tensor& in, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::softsign_spv, sizeof(shaders::softsign_spv));
				createPipeline(sizeof(SoftsignParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, out, 1, m_descriptor_set);
			SoftsignParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(SoftsignParam));
			runCommandBuffer();
			return true;
		}

		bool Softsign::computeGroupCount() {
			m_group_x =(int)alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}