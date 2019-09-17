#include <float.h>
#include "kernel.hpp"
#include "context.hpp"
#include <shaderc/shaderc.hpp>

namespace kernel {

	std::vector<uint32_t> compile(const std::string& name, shaderc_shader_kind kind, const std::string& data);
	void bindTensor(VkDevice& device, Tensor& tensor, int binding, VkDescriptorSet descriptor_set);
	/*
	void computeConvOutputShapeAndPadding(const PaddingMode& padding_mode,
		int& padding_top, int& padding_left,
		const int& in_h, const int& in_w,
		const int& filter_h, const int& filter_w,
		const int& dilation_h, const int& dilation_w,
		const int& stride_h, const int& stride_w,
		int& out_h, int& out_w);
	void computePoolOutputShape(const PaddingMode& padding_mode,
		const int& padding_top, const int& padding_left,
		const int& in_h, const int& in_w,
		const int& filter_h, const int& filter_w,
		const int& stride_h, const int& stride_w,
		int& out_h, int& out_w);
	*/

	inline bool checkFormat(Format fmt) { return fmt > -1 && fmt < kFormatNum; }
	inline size_t


}