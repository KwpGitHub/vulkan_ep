#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "sqrt.hpp"
#include <algorithm>

#define LOCAL_SZ_X 32
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct SqrtParam {
			int total;
		};

		Sqrt::Sqrt() {
			layer::initVulkanThing(2);
			m_type = "Sqrt";
		}

		void Sqrt::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Sqrt::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool Sqrt::forward(tensor& in, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::sqrt_spv, sizeof(shaders::sqrt_spv));
				createPipeline(sizeof(SqrtParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, out, 1, m_descriptor_set);
			SqrtParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(SqrtParam));
			runCommandBuffer();
			return true;
		}

		bool Sqrt::computeGroupCount() {
			m_group_x =(int)alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}