#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "ceil.hpp"
#include <algorithm>

#define LOCAL_SZ_X 32
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct CeilParam {
			int total;
		};

		Ceil::Ceil() {
			layer::initVulkanThing(2);
			m_type = "Ceil";
		}

		void Ceil::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Ceil::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool Ceil::forward(tensor& in, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::ceil_spv, sizeof(shaders::ceil_spv));
				createPipeline(sizeof(CeilParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, out, 1, m_descriptor_set);
			CeilParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(CeilParam));
			runCommandBuffer();
			return true;
		}

		bool Ceil::computeGroupCount() {
			m_group_x = alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}