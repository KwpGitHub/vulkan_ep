#ifndef GLOBALMAXPOOL_H
#define GLOBALMAXPOOL_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class GlobalMaxPool : public Layer {
    public:
        GlobalMaxPool() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~GlobalMaxPool(){}

    };
}

#endif
