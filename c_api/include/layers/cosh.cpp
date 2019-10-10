#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "cosh.hpp"
#include <algorithm>

#define LOCAL_SZ_X 32
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct CoshParam {
			int total;
		};

		Cosh::Cosh() {
			layer::initVulkanThing(2);
			m_type = "Cosh";
		}

		void Cosh::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Cosh::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool Cosh::forward(tensor& in, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::tanh_spv, sizeof(shaders::tanh_spv));
				createPipeline(sizeof(CoshParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, out, 1, m_descriptor_set);
			CoshParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(CoshParam));
			runCommandBuffer();
			return true;
		}

		bool Cosh::computeGroupCount() {
			m_group_x =(int)alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}