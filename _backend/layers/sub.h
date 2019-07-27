#ifndef SUB_H
#define SUB_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Sub : public Layer {
    public:
        Sub() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Sub(){}

    };
}

#endif
