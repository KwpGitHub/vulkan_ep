#ifndef TREEENSEMBLEREGRESSOR_H
#define TREEENSEMBLEREGRESSOR_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class TreeEnsembleRegressor : public Layer {
    public:
        TreeEnsembleRegressor() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~TreeEnsembleRegressor(){}

    };
}

#endif
