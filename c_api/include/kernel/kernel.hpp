#pragma once
#include <vector>

namespace kernel {
	enum Format {
		kFormatInvalid = -1,
		kFormatFp16,
		kFormatFp32,
		kFormatFp64,
		kFormatInt32,
		kFormatNum
	};

	enum OpType {
		kOpTypeConv,
		kOpTypePool,
		kOpTypeDWConv,
		kOpTypeLRN,
		kOpTypeConcat,
		kOpTypeSoftmax,
		kOpTypeReLU,
		kOpTypePriorBox,
		kOpTypePermute,
		kOpTypeNum
	};

	enum PaddingMode { kPaddingModeSame, kPaddingModeValid, kPaddingModeCaffe, kPaddingModeNum };
	enum FusedActivationType { kNone, kRelu, kRelu1, kRelu6, kActivationNum };
	typedef std::vector<int> Shape;

	bool isAvailable();
}

#include "kernel/tensor.hpp"
#include "kernel/buffer.hpp"
#include "kernel/layer.hpp"