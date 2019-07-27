#ifndef ONEHOTENCODER_H
#define ONEHOTENCODER_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class OneHotEncoder : public Layer {
    public:
        OneHotEncoder() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~OneHotEncoder(){}

    };
}

#endif
