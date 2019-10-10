#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "mod.hpp"
#include <algorithm>

#define LOCAL_SZ_X 32
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct ModParam {
			int total;
		};

		Mod::Mod() {
			layer::initVulkanThing(2);
			m_type = "Mod";
		}

		void Mod::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Mod::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], ins[1], outs[0]);
		}

		bool Mod::forward(tensor& in, tensor& in2, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::mod_spv, sizeof(shaders::mod_spv));
				createPipeline(sizeof(ModParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, in2, 1, m_descriptor_set);
			bindTensor(m_device, out, 2, m_descriptor_set);
			ModParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(ModParam));
			runCommandBuffer();
			return true;
		}

		bool Mod::computeGroupCount() {
			m_group_x =(int)alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}