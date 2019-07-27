#ifndef CONVTRANSPOSE_H
#define CONVTRANSPOSE_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ConvTranspose : public Layer {
    public:
        ConvTranspose() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~ConvTranspose(){}

    };
}

#endif
