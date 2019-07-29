#ifndef ZIPMAP_H
#define ZIPMAP_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class ZipMap : public Layer {
    public:
        ZipMap() {
        }
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~ZipMap(){}

    };
}

#endif
