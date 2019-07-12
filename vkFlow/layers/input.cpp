#include "input.h"

namespace backend {
	namespace CPU {
		Input::Input() {
			one_blob_only = true;
			support_inplace = true;
			support_vulkan = false;
		}

		int Input::forward_inplace(Mat& /*bottom_top_blob*/, const Option& /*opt*/) const
		{
			return 0;
		}
	}
	namespace GPU {

	}
}