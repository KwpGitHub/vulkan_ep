#ifndef LAYERREPO_H
#define LAYERREOP_H
#include <vector>

namespace layer {
#include "layer.h"
	std::vector<backend::Layer*> model_layers;
}

namespace layer{
#include "./layers/absval.h"
	class AbsVal_t {
		char use_gpu = 0;
	public:
		AbsVal_t() {
			backend::CPU::AbsVal* t = new backend::CPU::AbsVal();
			model_layers.push_back(t);
		}
	};

}

#endif