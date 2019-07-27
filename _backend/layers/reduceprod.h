#ifndef REDUCEPROD_H
#define REDUCEPROD_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReduceProd : public Layer {
    public:
        ReduceProd() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~ReduceProd(){}

    };
}

#endif
