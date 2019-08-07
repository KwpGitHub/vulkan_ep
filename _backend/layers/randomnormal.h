#ifndef RANDOMNORMAL_H
#define RANDOMNORMAL_H //RandomNormal

//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               shape
//PARAMETER_TYPES:          INTS
//OPTIONAL_PARAMETERS:      dtype, mean, scale, seed
//OPTIONAL_PARAMETERS_TYPE: INT, FLOAT, FLOAT, FLOAT

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class RandomNormal : public Layer {
        
        vuh::Device* _get_device();

        struct Params{ };
        vuh::Program<Specs, Params>* program;

    public:
        RandomNormal(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
         
         //std::vector<uint32_t> output_shape();
   
        ~RandomNormal(){}
    };
}


namespace backend {    
    RandomNormal::RandomNormal(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/randomnormal.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* RandomNormal::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }


};

#endif
