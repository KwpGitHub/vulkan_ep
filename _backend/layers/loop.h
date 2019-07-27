#ifndef LOOP_H
#define LOOP_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Loop : public Layer {
    public:
        Loop() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Loop(){}

    };
}

#endif
