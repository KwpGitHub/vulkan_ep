#ifndef LOG_H
#define LOG_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Log : public Layer {
    public:
        Log() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Log(){}

    };
}

#endif
