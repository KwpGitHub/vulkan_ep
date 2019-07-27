#ifndef BINARIZER_H
#define BINARIZER_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Binarizer : public Layer {
    public:
        Binarizer() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Binarizer(){}

    };
}

#endif
