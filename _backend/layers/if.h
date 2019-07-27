#ifndef IF_H
#define IF_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class If : public Layer {
    public:
        If() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~If(){}

    };
}

#endif
