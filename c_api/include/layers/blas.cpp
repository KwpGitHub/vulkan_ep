#include "kernel/common.hpp"
#include "kernel/utils.hpp"
#include "blas.hpp"
#include <algorithm>

#define LOCAL_SZ_X 32
#define maxComputeWorkGroupCount 65535

namespace kernel {
	namespace layers {
		struct GemmParam {
			int M;
            int N;
            int K;
            bool use_bias;
		};

		Gemm::Gemm() {
			layer::initVulkanThing(2);
			m_type = "Gemm";
		}

		void Gemm::reshapeOutTensor(tensor& in, tensor& out) {
			Shape shape = in.getShape();
			out = out.reshape(nullptr, shape);
		}

		bool Gemm::forward(std::vector<tensor>& ins, std::vector<tensor>& blobs, std::vector<tensor>& outs) {
			return forward(ins[0], ins[1], ins[2], outs[0]);
		}

		bool Gemm::forward(tensor& A,tenosr& B, tensor& C, tensor& D) {
			if (m_pipeline == VK_NULL_HANDLE) {
				m_total = in.count();
				computeGroupCount();
				createShaderModule(shaders::gemm_spv, sizeof(shaders::gemm_spv));
				createPipeline(sizeof(GemmParam));
			}

			bindTensor(m_device, A, 0, m_descriptor_set);
            bindTensor(m_device, B, 1, m_descriptor_set);
			bindTensor(m_device, C, 2, m_descriptor_set);
            bindTensor(m_device, D, 3, m_descriptor_set);
            GemmParam param = { M, N, K, use_bias };
			recordCommandBuffer((void*)& param, sizeof(GemmParam));
			runCommandBuffer();
			return true;
		}

		bool Gemm::computeGroupCount() {
			m_group_x = alignSize(m_total, LOCAL_SZ_X) / LOCAL_SZ_X;
			if (m_group_x > maxComputeWorkGroupCount)
				m_group_x = maxComputeWorkGroupCount;
			m_group_y = 1;
			m_group_z = 1;
			return true;
		}
	}
}