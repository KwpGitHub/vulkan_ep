#ifndef IMPUTER_H
#define IMPUTER_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Imputer : public Layer {
    public:
        Imputer() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Imputer(){}

    };
}

#endif
