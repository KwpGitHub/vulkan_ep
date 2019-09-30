#pragma once
#include <vector>

namespace kernel {
	enum Format {
		kFormatInvalid = -1,
		kFormatFp16,
		kFormatFp32,
		kFormatFp64,
		kFormatInt32,
		kFormatBool,
		kFormatNum
	};

	enum PaddingMode { kPaddingModeSame, kPaddingModeValid, kPaddingModeNum };
	enum FusedActivationType { kNone, kRelu, kRelu1, kRelu6, kActivationNum };
	typedef std::vector<int> Shape;

	bool isAvailable();
}

#include "kernel/tensor.hpp"
#include "kernel/buffer.hpp"
#include "kernel/layer.hpp"




#include "layers/abs.hpp"
#include "layers/acos.hpp"
#include "layers/acosh.hpp"
#include "layers/add.hpp"
#include "layers/and.hpp"
#include "layers/asin.hpp"
#include "layers/asinh.hpp"
#include "layers/atan.hpp"
#include "layers/atanh.hpp"

#include "layers/ceil.hpp"
#include "layers/clip.hpp"

#include "layers/elu.hpp"
#include "layers/equal.hpp"
#include "layers/exp.hpp"

#include "layers/floor.hpp"

#include "layers/greater.hpp"

#include "layers/hardsigmoid.hpp"

#include "layers/leakyrelu.hpp"
#include "layers/less.hpp"
#include "layers/log.hpp"

#include "layers/max.hpp"
#include "layers/min.hpp"
#include "layers/mod.hpp"
#include "layers/mul.hpp"

#include "layers/neg.hpp"
#include "layers/not.hpp"

#include "layers/or.hpp"

#include "layers/pow.hpp"
#include "layers/prelu.hpp"

#include "layers/reciprocal.hpp"
#include "layers/relu.hpp"
#include "layers/round.hpp"

#include "layers/selu.hpp"
#include "layers/sigmoid.hpp"
#include "layers/sin.hpp"
#include "layers/sinh.hpp"
#include "layers/softplus.hpp"
#include "layers/softsign.hpp"
#include "layers/sqrt.hpp"
#include "layers/sub.hpp"
#include "layers/sum.hpp"

#include "layers/tanh.hpp"

#include "layers/xor.hpp"