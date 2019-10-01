#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "pow.hpp"
#include <algorithm>

#define LOCAL_SZ_X 32
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct PowParam {
			int total;
		};

		Pow::Pow() {
			layer::initVulkanThing(2);
			m_type = "Pow";
		}

		void Pow::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Pow::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0],ins[2], outs[0]);
		}

		bool Pow::forward(tensor& in, tensor& in2, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::pow_spv, sizeof(shaders::pow_spv));
				createPipeline(sizeof(PowParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, in2, 1, m_descriptor_set);
			bindTensor(m_device, out, 2, m_descriptor_set);
			PowParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(PowParam));
			runCommandBuffer();
			return true;
		}

		bool Pow::computeGroupCount() {
			m_group_x = alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}