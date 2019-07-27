#ifndef REDUCEMEAN_H
#define REDUCEMEAN_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReduceMean : public Layer {
    public:
        ReduceMean() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~ReduceMean(){}

    };
}

#endif
