#include "unaryop.h"

namespace backend {
	namespace CPU {
		UnaryOp::UnaryOp()
		{
			one_blob_only = true;
			support_inplace = true;
		}
	}
	namespace GPU {

	}
}