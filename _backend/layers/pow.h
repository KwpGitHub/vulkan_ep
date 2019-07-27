#ifndef POW_H
#define POW_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Pow : public Layer {
    public:
        Pow() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Pow(){}

    };
}

#endif
