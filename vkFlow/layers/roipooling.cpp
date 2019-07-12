#include "roipooling.h"
#include <algorithm>

namespace backend {
	namespace CPU {
		ROIPooling::ROIPooling() {}

		int ROIPooling::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const {
			const Mat& bottom_blob = bottom_blobs[0];
			int w = bottom_blob.w;
			int h = bottom_blob.h;
			size_t elemsize = bottom_blob.elemsize;
			int channels = bottom_blob.c;
			const Mat& roi_blob = bottom_blobs[1];
			Mat& top_blob = top_blobs[0];
			top_blob.create(pooled_width, pooled_height, channels, elemsize, opt.blob_allocator);
			if (top_blob.empty())
				return -100;

			const float* roi_ptr = roi_blob;

			int roi_x1 = round(roi_ptr[0] * spatial_scale);
			int roi_y1 = round(roi_ptr[1] * spatial_scale);
			int roi_x2 = round(roi_ptr[2] * spatial_scale);
			int roi_y2 = round(roi_ptr[3] * spatial_scale);

			int roi_w = std::max<int>(roi_x2 - roi_x1 + 1, 1);
			int roi_h = std::max<int>(roi_y2 - roi_y1 + 1, 1);

			float bin_size_w = (float)roi_w / (float)pooled_width;
			float bin_size_h = (float)roi_h / (float)pooled_height;

#pragma omp parallel for num_threads(opt.num_threads)
			for (int q = 0; q < channels; q++) {
				const float* ptr = bottom_blob.channel(q);
				float* outptr = top_blob.channel(q);

				for (int ph = 0; ph < pooled_height; ph++) {
					for (int pw = 0; pw < pooled_width; pw++) {						
						int hstart = roi_y1 + floor((float)(ph)* bin_size_h);
						int wstart = roi_x1 + floor((float)(pw)* bin_size_w);
						int hend = roi_y1 + ceil((float)(ph + 1) * bin_size_h);
						int wend = roi_x1 + ceil((float)(pw + 1) * bin_size_w);

						hstart = std::min<int>(std::max<int>(hstart, 0), h);
						wstart = std::min<int>(std::max<int>(wstart, 0), w);
						hend = std::min<int>(std::max<int>(hend, 0), h);
						wend = std::min<int>(std::max<int>(wend, 0), w);

						bool is_empty = (hend <= hstart) || (wend <= wstart);
						float max = is_empty ? 0.f : ptr[hstart * w + wstart];

						for (int y = hstart; y < hend; y++)	{
							for (int x = wstart; x < wend; x++) {
								int index = y * w + x;
								max = std::max<float>(max, ptr[index]);
							}
						}

						outptr[pw] = max;
					}

					outptr += pooled_width;
				}
			}

			return 0;
		}
	}
	namespace GPU {

	}
}