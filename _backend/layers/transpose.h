#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Transpose : public Layer {
    public:
        Transpose() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Transpose(){}

    };
}

#endif
