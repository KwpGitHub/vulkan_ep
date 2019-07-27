#ifndef NORMALIZER_H
#define NORMALIZER_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Normalizer : public Layer {
    public:
        Normalizer() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Normalizer(){}

    };
}

#endif
