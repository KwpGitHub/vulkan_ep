#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "mul.hpp"
#include <algorithm>

#define LOCAL_SZ_X 32
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct MulParam {
			int total;
		};

		Mul::Mul() {
			layer::initVulkanThing(2);
			m_type = "Mul";
		}

		void Mul::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Mul::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], ins[1], outs[0]);
		}

		bool Mul::forward(tensor& in,  tensor& in2, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::mul_spv, sizeof(shaders::mul_spv));
				createPipeline(sizeof(MulParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, in2, 1, m_descriptor_set);
			bindTensor(m_device, out, 2, m_descriptor_set);
			MulParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(MulParam));
			runCommandBuffer();
			return true;
		}

		bool Mul::computeGroupCount() {
			m_group_x =(int)alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}