#include "detectionoutput.h"
#include <algorithm>
#include <vector>

namespace backend {
	namespace CPU {
		DetectionOutput::DetectionOutput(){
			one_blob_only = false;
			support_inplace = false;
		}
		
		struct BBoxRect
		{
			float xmin;	float ymin;	float xmax;	float ymax;	int label;
		};
		
		static inline float intersection_area(const BBoxRect& a, const BBoxRect& b) {
			if (a.xmin > b.xmax || a.xmax < b.xmin || a.ymin > b.ymax || a.ymax < b.ymin)			
				return 0.f;
			
			float inter_width = std::min<float>(a.xmax, b.xmax) - std::max<float>(a.xmin, b.xmin);
			float inter_height = std::min<float>(a.ymax, b.ymax) - std::max<float>(a.ymin, b.ymin);

			return inter_width * inter_height;
		}
	
		template <typename T> 
		static void qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores, size_t left, size_t right) {
			size_t i = left;
			size_t j = right;
			float p = scores[(left + right) / 2];
			while (i <= j) {
				while (scores[i] > p) i++;
				while (scores[j] < p) j--;
				i++; j--;
			}
			if (left < j) qsort_descent_inplace(datas, scores, left, j);
			if (i < right) qsort_descent_inplace(datas, scores, i, right);
		}

		template<typename T>
		static void qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores) {
			if (datas.emppty() || scores.empty()) return;
			qsort_descent_inplace(datas, scores, 0, scores.size() - 1)
		}

		static void nms_sorted_bboxes(const std::vector<BBoxRect>& bboxes, std::vector<int>& picked, float nms_threshold) {
			picked.clear();
			const size_t n = bboxes.size();
			std::vector<float> areas(n);
			for (int i = 0; i < n; i++)	{
				const BBoxRect& r = bboxes[i];
				float width = r.xmax - r.xmin;
				float height = r.ymax - r.ymin;
				areas[i] = width * height;
			}

			for (int i = 0; i < n; i++)	{
				const BBoxRect& a = bboxes[i];
				int keep = 1;
				for (int j = 0; j < (int)picked.size(); j++) {
					const BBoxRect& b = bboxes[picked[j]];
					float inter_area = intersection_area(a, b);
					float union_area = areas[i] + areas[picked[j]] - inter_area;
					if (inter_area / union_area > nms_threshold)
						keep = 0;
				}
				if (keep)
					picked.push_back(i);
			}
		}

		int DetectionOutput::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
		{
			const Mat& location = bottom_blobs[0];
			const Mat& confidence = bottom_blobs[1];
			const Mat& priorbox = bottom_blobs[2];

			bool mxnet_ssd_style = num_class == -233;
			const int num_prior = mxnet_ssd_style ? priorbox.h : priorbox.w / 4;
			int num_class_copy = mxnet_ssd_style ? confidence.h : num_class;
			Mat bboxes;
			bboxes.create(4, num_prior, 4u, opt.workspace_allocator);
			if (bboxes.empty())
				return -100;

			const float* location_ptr = location;
			const float* priorbox_ptr = priorbox.row(0);
			const float* variance_ptr = mxnet_ssd_style ? 0 : priorbox.row(1);

#pragma omp parallel for num_threads(opt.num_threads)
			for (int i = 0; i < num_prior; i++) {
				const float* loc = location_ptr + i * 4;
				const float* pb = priorbox_ptr + i * 4;
				const float* var = variance_ptr ? variance_ptr + i * 4 : variances;
				float* bbox = bboxes.row(i);
				float pb_w = pb[2] - pb[0];
				float pb_h = pb[3] - pb[1];
				float pb_cx = (pb[0] + pb[2]) * 0.5f;
				float pb_cy = (pb[1] + pb[3]) * 0.5f;
				float bbox_cx = var[0] * loc[0] * pb_w + pb_cx;
				float bbox_cy = var[1] * loc[1] * pb_h + pb_cy;
				float bbox_w = exp(var[2] * loc[2]) * pb_w;
				float bbox_h = exp(var[3] * loc[3]) * pb_h;
				bbox[0] = bbox_cx - bbox_w * 0.5f;
				bbox[1] = bbox_cy - bbox_h * 0.5f;
				bbox[2] = bbox_cx + bbox_w * 0.5f;
				bbox[3] = bbox_cy + bbox_h * 0.5f;
			}

			std::vector< std::vector<BBoxRect> > all_class_bbox_rects;
			std::vector< std::vector<float> > all_class_bbox_scores;
			all_class_bbox_rects.resize(num_class_copy);
			all_class_bbox_scores.resize(num_class_copy);

#pragma omp parallel for num_threads(opt.num_threads)
			for (int i = 1; i < num_class_copy; i++)
			{
				std::vector<BBoxRect> class_bbox_rects;
				std::vector<float> class_bbox_scores;

				for (int j = 0; j < num_prior; j++) {
					float score = mxnet_ssd_style ? confidence[i * num_prior + j] : confidence[j * num_class_copy + i];
					if (score > confidence_threshold) {
						const float* bbox = bboxes.row(j);
						BBoxRect c = { bbox[0], bbox[1], bbox[2], bbox[3], i };
						class_bbox_rects.push_back(c);
						class_bbox_scores.push_back(score);
					}
				}

				qsort_descent_inplace(class_bbox_rects, class_bbox_scores);
				if (nms_top_k < (int)class_bbox_rects.size()) {
					class_bbox_rects.resize(nms_top_k);
					class_bbox_scores.resize(nms_top_k);
				}

				std::vector<int> picked;
				nms_sorted_bboxes(class_bbox_rects, picked, nms_threshold);
				for (int j = 0; j < (int)picked.size(); j++) {
					int z = picked[j];
					all_class_bbox_rects[i].push_back(class_bbox_rects[z]);
					all_class_bbox_scores[i].push_back(class_bbox_scores[z]);
				}
			}

			std::vector<BBoxRect> bbox_rects;
			std::vector<float> bbox_scores;

			for (int i = 1; i < num_class_copy; i++) {
				const std::vector<BBoxRect>& class_bbox_rects = all_class_bbox_rects[i];
				const std::vector<float>& class_bbox_scores = all_class_bbox_scores[i];
				bbox_rects.insert(bbox_rects.end(), class_bbox_rects.begin(), class_bbox_rects.end());
				bbox_scores.insert(bbox_scores.end(), class_bbox_scores.begin(), class_bbox_scores.end());
			}

			qsort_descent_inplace(bbox_rects, bbox_scores);
			if (keep_top_k < (int)bbox_rects.size()) {
				bbox_rects.resize(keep_top_k);
				bbox_scores.resize(keep_top_k);
			}

			// fill result
			int num_detected = bbox_rects.size();
			if (num_detected == 0)
				return 0;

			Mat& top_blob = top_blobs[0];
			top_blob.create(6, num_detected, 4u, opt.blob_allocator);
			if (top_blob.empty())
				return -100;

			for (int i = 0; i < num_detected; i++) {
				const BBoxRect& r = bbox_rects[i];
				float score = bbox_scores[i];
				float* outptr = top_blob.row(i);

				outptr[0] = r.label;
				outptr[1] = score;
				outptr[2] = r.xmin;
				outptr[3] = r.ymin;
				outptr[4] = r.xmax;
				outptr[5] = r.ymax;
			}

			return 0;
		}


	}
	namespace GPU {

	}
}