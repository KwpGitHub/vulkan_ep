#ifndef GLOBALLPPOOL_H
#define GLOBALLPPOOL_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class GlobalLpPool : public Layer {
    public:
        GlobalLpPool() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~GlobalLpPool(){}

    };
}

#endif
