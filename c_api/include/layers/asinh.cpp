#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "asinh.hpp"
#include <algorithm>

#define LOCAL_SZ_X 1024
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct AsinhParam {
			int total;
		};

		Asinh::Asinh() {
			layer::initVulkanThing(2);
			m_type = "Asinh";
		}

		void Asinh::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Asinh::forward(std::vector<tensor>& ins, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool Asinh::forward(tensor& in, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::asinh_spv, sizeof(shaders::asinh_spv));
				createPipeline(sizeof(AsinhParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, out, 1, m_descriptor_set);
			AsinhParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(AsinhParam));
			runCommandBuffer();
			return true;
		}

		bool Asinh::computeGroupCount() {
			m_group_x = (int)alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}