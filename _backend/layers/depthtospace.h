#ifndef DEPTHTOSPACE_H
#define DEPTHTOSPACE_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class DepthToSpace : public Layer {
    public:
        DepthToSpace(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {}
        
        vuh::Array<float>& operator()(const vuh::Array<float>& t) {
            
        }

        void forward(){
        
        }

        void build_pipeline(){
            
        }

        ~DepthToSpace(){}

    };
}

#endif
