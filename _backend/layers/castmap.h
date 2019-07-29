#ifndef CASTMAP_H
#define CASTMAP_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class CastMap : public Layer {
    public:
        CastMap() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~CastMap(){}

    };
}

#endif
