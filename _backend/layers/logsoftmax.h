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
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~LogSoftmax(){}

    };
}

#endif
