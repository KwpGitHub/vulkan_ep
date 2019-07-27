#ifndef COS_H
#define COS_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Cos : public Layer {
    public:
        Cos() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Cos(){}

    };
}

#endif
