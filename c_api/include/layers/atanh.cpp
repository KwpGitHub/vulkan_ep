#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "atanh.hpp"
#include <algorithm>

#define LOCAL_SZ_X 32
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct AtanhParam {
			int total;
		};

		Atanh::Atanh() {
			layer::initVulkanThing(2);
			m_type = "Atanh";
		}

		void Atanh::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Atanh::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool Atanh::forward(tensor& in, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::atanh_spv, sizeof(shaders::atanh_spv));
				createPipeline(sizeof(AtanhParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, out, 1, m_descriptor_set);
			AtanhParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(AtanhParam));
			runCommandBuffer();
			return true;
		}

		bool Atanh::computeGroupCount() {
			m_group_x = (int)alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}