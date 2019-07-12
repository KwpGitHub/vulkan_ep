#include "argmax.h"
#include <algorithm>
#include <functional>

namespace backend {
	namespace CPU {

		ArgMax::ArgMax() { one_blob_only = true; }

		int ArgMax::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const {
			int size = bottom_blob.total();

			if (out_max_val)
				top_blob.create(topk, 2, 4u, opt.blob_allocator);
			else
				top_blob.create(topk, 1, 4u, opt.blob_allocator);
			if (top_blob.empty())
				return -100;

			const float* ptr = bottom_blob;

			std::vector< std::pair<float, int> > vec;
			vec.resize(size);
			for (int i = 0; i < size; i++)
				vec[i] = std::make_pair(ptr[i], i);
	
			std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(), std::greater< std::pair<float, int> >());

			float* outptr = top_blob;
			if (out_max_val)
			{
				float* valptr = outptr + topk;
				for (int i = 0; i < topk; i++)
				{
					outptr[i] = vec[i].first;
					valptr[i] = vec[i].second;
				}
			}
			else
			{
				for (int i = 0; i < topk; i++)
					outptr[i] = vec[i].second;
				
			}

			return 0;
		}
	}
	namespace GPU {

	}
}
