#include "layer.h"
#include "layers/absval.h"
#include <vector>

namespace backend{
	std::vector<Layer*> model_layers;

	class AbsVal_t {
	public:
		AbsVal_t() {
			CPU::AbsVal* t = new CPU::AbsVal();
			model_layers.push_back(t);
		}
	};





}
