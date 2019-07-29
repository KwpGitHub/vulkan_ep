#ifndef REDUCEMEAN_H
#define REDUCEMEAN_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReduceMean : public Layer {
    public:
        ReduceMean() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~ReduceMean(){}

    };
}

#endif
