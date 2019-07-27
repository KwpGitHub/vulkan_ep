#ifndef SCALER_H
#define SCALER_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Scaler : public Layer {
    public:
        Scaler() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Scaler(){}

    };
}

#endif
