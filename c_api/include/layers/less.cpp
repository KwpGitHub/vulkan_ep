#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "less.hpp"
#include <algorithm>

#define LOCAL_SZ_X 1024
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct LessParam {
			int total;
		};

		Less::Less() {
			layer::initVulkanThing(2);
			m_type = "Less";
		}

		void Less::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Less::forward(std::vector<tensor>& ins, std::vector<tensor>& outs) {
			return forward(ins[0], ins[1], outs[0]);
		}

		bool Less::forward(tensor& in, tensor& in2, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::less_spv, sizeof(shaders::less_spv));
				createPipeline(sizeof(LessParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, in2, 1, m_descriptor_set);
			bindTensor(m_device, out, 2, m_descriptor_set);
			LessParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(LessParam));
			runCommandBuffer();
			return true;
		}

		bool Less::computeGroupCount() {
			m_group_x =(int)alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}