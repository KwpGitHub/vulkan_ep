#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "floor.hpp"
#include <algorithm>

#define LOCAL_SZ_X 1024
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct FloorParam {
			int total;
		};

		Floor::Floor() {
			layer::initVulkanThing(2);
			m_type = "Floor";
		}

		void Floor::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Floor::forward(std::vector<tensor>& ins, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool Floor::forward(tensor& in, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::floor_spv, sizeof(shaders::floor_spv));
				createPipeline(sizeof(FloorParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, out, 1, m_descriptor_set);
			FloorParam param = { m_total };
			recordCommandBuffer((void*)& param, sizeof(FloorParam));
			runCommandBuffer();
			return true;
		}

		bool Floor::computeGroupCount() {
			m_group_x =(int)alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}