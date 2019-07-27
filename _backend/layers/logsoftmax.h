#ifndef LOGSOFTMAX_H
#define LOGSOFTMAX_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class LogSoftmax : public Layer {
    public:
        LogSoftmax() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~LogSoftmax(){}

    };
}

#endif
