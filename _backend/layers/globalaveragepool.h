#ifndef GLOBALAVERAGEPOOL_H
#define GLOBALAVERAGEPOOL_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class GlobalAveragePool : public Layer {
    public:
        GlobalAveragePool() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~GlobalAveragePool(){}

    };
}

#endif
