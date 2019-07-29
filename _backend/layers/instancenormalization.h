#ifndef INSTANCENORMALIZATION_H
#define INSTANCENORMALIZATION_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class InstanceNormalization : public Layer {
    public:
        InstanceNormalization() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~InstanceNormalization(){}

    };
}

#endif
