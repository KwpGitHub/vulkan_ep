#ifndef ADD_H
#define ADD_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Add : public Layer {
    public:
        Add() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Add(){}

    };
}

#endif
