#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "greater.hpp"
#include <algorithm>

#define LOCAL_SZ_X 32
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct GreaterParam {
			int total;
		};

		Greater::Greater() {
			layer::initVulkanThing(2);
			m_type = "Greater";
		}

		void Greater::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Greater::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], ins[1], outs[0]);
		}

		bool Greater::forward(tensor& in, tensor& in2, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::greater_spv, sizeof(shaders::greater_spv));
				createPipeline(sizeof(GreaterParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, in2, 1, m_descriptor_set);
			bindTensor(m_device, out, 2, m_descriptor_set);
			GreaterParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(GreaterParam));
			runCommandBuffer();
			return true;
		}

		bool Greater::computeGroupCount() {
			m_group_x =(int)alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}