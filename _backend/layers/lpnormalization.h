#ifndef LPNORMALIZATION_H
#define LPNORMALIZATION_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class LpNormalization : public Layer {
    public:
        LpNormalization() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~LpNormalization(){}

    };
}

#endif
