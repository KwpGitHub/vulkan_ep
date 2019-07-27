#ifndef CONCAT_H
#define CONCAT_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Concat : public Layer {
    public:
        Concat() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Concat(){}

    };
}

#endif
