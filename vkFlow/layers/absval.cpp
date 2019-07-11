#include "absval.h"

namespace backend {
	namespace CPU{

		AbsVal::AbsVal() {
			one_blob_only = true;
			support_inplace = true;		
		}

		int AbsVal::forward_inplace(Mat& bottom_top_blob, const Option& opt) const {
			int w = bottom_top_blob.w;
			int h = bottom_top_blob.h;
			int channels = bottom_top_blob.c;
			int size = w * h;

#pragma omp parallel for num_threads(opt.num_threads)
			for (int q = 0; q < channels; q++)
			{
				float* ptr = bottom_top_blob.channel(q);
				for (int i = 0; i < size; i++)
				{
					if (ptr[i] < 0)
						ptr[i] = -ptr[i];
				}
			}

			return 0;
		}

	}
	
	namespace GPU {
		AbsVal::AbsVal() {
			support_vulkan = true;
			pipeline_absval = 0;
			pipeline_absval_pack4 = 0;
		}

		int AbsVal::create_pipeline(const Option& opt)
		{
			std::vector<vk_specialization_type> specializations;

			// pack1
			{
				pipeline_absval = new Pipeline(vkdev);
				pipeline_absval->set_optimal_local_size_xyz();
				pipeline_absval->create("absval", opt, specializations, 1, 5);
			}

			// pack4
			{
				pipeline_absval_pack4 = new Pipeline(vkdev);
				pipeline_absval_pack4->set_optimal_local_size_xyz();
				pipeline_absval_pack4->create("absval_pack4", opt, specializations, 1, 5);
			}

			return 0;
		}

		int AbsVal::destroy_pipeline(const Option& opt)
		{
			delete pipeline_absval;
			pipeline_absval = 0;

			delete pipeline_absval_pack4;
			pipeline_absval_pack4 = 0;

			return 0;
		}

		int AbsVal::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
		{
			int packing = bottom_top_blob.packing;

			std::vector<VkMat> bindings(1);
			bindings[0] = bottom_top_blob;

			std::vector<vk_constant_type> constants(5);
			constants[0].i = bottom_top_blob.dims;
			constants[1].i = bottom_top_blob.w;
			constants[2].i = bottom_top_blob.h;
			constants[3].i = bottom_top_blob.c;
			constants[4].i = bottom_top_blob.cstep;

			const Pipeline* pipeline = packing == 4 ? pipeline_absval_pack4 : pipeline_absval;

			cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);

			return 0;
		}

	}

}