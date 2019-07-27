#ifndef TILE_H
#define TILE_H

#include <vector>
#include "../layer.h"
#include "../tensor.h"
#include "../kernel/vuh.h"

namespace backend {
    class Tile : public Layer {
    public:
        Tile() {
        }
        
        Tensor& operator()(const Tensor& t) {
        }

        void forward(){          
        }

        ~Tile(){}

    };
}

#endif
