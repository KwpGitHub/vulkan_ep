#ifndef INSTANCENORMALIZATION_H
#define INSTANCENORMALIZATION_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class InstanceNormalization : public Layer {
    public:
        InstanceNormalization() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~InstanceNormalization(){}

    };
}

#endif
