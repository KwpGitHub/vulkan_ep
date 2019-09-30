#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "clip.hpp"
#include <algorithm>

#define LOCAL_SZ_X 32
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct ClipParam {
			int total;
			float max;
			float min;
		};

		Clip::Clip(float _max, float _min) : m_max(_max), m_min(_min) {
			layer::initVulkanThing(2);
			m_type = "Clip";			
		}

		void Clip::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Clip::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], outs[0]);
		}

		bool Clip::forward(tensor& in, tensor& out) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::clip_spv, sizeof(shaders::clip_spv));
				createPipeline(sizeof(ClipParam));
			}

			bindTensor(m_device, in, 0, m_descriptor_set);
			bindTensor(m_device, out, 1, m_descriptor_set);
			ClipParam param = { m_total, m_max, m_min };
			recordCommandBuffer((void*)& param, sizeof(ClipParam));
			runCommandBuffer();
			return true;
		}

		bool Clip::computeGroupCount() {
			m_group_x = alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}