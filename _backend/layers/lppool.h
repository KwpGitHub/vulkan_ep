#ifndef LPPOOL_H
#define LPPOOL_H //LpPool

//INPUTS:                   X
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               kernel_shape
//PARAMETER_TYPES:          INTS
//OPTIONAL_PARAMETERS:      auto_pad, p, pads, strides
//OPTIONAL_PARAMETERS_TYPE: STRING, INT, INTS, INTS

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class LpPool : public Layer {
        
        vuh::Device* _get_device();

        struct Params{ };
        vuh::Program<Specs, Params>* program;

    public:
        LpPool(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
         
         //std::vector<uint32_t> output_shape();
   
        ~LpPool(){}
    };
}


namespace backend {    
    LpPool::LpPool(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/lppool.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* LpPool::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }


};

#endif
