#ifndef LSTM_H
#define LSTM_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class LSTM : public Layer {
    public:
        LSTM() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~LSTM(){}

    };
}

#endif
