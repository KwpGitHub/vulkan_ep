#ifndef MAXROIPOOL_H
#define MAXROIPOOL_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class MaxRoiPool : public Layer {
    public:
        MaxRoiPool() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~MaxRoiPool(){}

    };
}

#endif
