#ifndef AND_H
#define AND_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class And : public Layer {
    public:
        And() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~And(){}

    };
}

#endif
