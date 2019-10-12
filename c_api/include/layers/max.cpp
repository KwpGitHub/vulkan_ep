#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "max.hpp"
#include <algorithm>

#define LOCAL_SZ_X 1024
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct MaxParam {
			int total;
		};

		Max::Max() {
			layer::initVulkanThing(2);
			m_type = "Max";
		}

		void Max::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Max::forward(std::vector<tensor>& ins, std::vector<tensor>& outs) {
			return forward(ins[0], ins[1], outs[0]);
		}

		bool Max::forward(tensor& in, tensor& in2, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::max_spv, sizeof(shaders::max_spv));
				createPipeline(sizeof(MaxParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, in2, 1, m_descriptor_set);
			bindTensor(m_device, out, 2, m_descriptor_set);
			MaxParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(MaxParam));
			runCommandBuffer();
			return true;
		}

		bool Max::computeGroupCount() {
			m_group_x =(int)alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}