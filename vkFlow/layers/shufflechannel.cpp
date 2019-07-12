#include "shufflechannel.h"

namespace backend {
	namespace CPU {
		ShuffleChannel::ShuffleChannel() {
			one_blob_only = true;
			support_inplace = false;
		}
		
		int ShuffleChannel::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const	{
			int w = bottom_blob.w;
			int h = bottom_blob.h;
			int c = bottom_blob.c;
			size_t elemsize = bottom_blob.elemsize;
			int chs_per_group = c / group;

			if (c != chs_per_group * group)	{
				return -100;
			}

			top_blob.create(w, h, c, elemsize, opt.blob_allocator);
			if (top_blob.empty())
				return -100;

			const size_t feature_sz = w * h * elemsize;
			for (int i = 0; i != group; i++) {
				for (int j = 0; j != chs_per_group; j++) {
					int src_q = chs_per_group * i + j;
					int dst_q = group * j + i;
					memcpy(top_blob.channel(dst_q), bottom_blob.channel(src_q), feature_sz);
				}
			}

			return 0;
		}

	}
	namespace GPU {

	}
}