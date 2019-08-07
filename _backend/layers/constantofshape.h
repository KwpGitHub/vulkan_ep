#ifndef CONSTANTOFSHAPE_H
#define CONSTANTOFSHAPE_H //ConstantOfShape

//INPUTS:                   input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      value
//OPTIONAL_PARAMETERS_TYPE: TENSOR

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class ConstantOfShape : public Layer {
        
        vuh::Device* _get_device();

        struct Params{ };
        vuh::Program<Specs, Params>* program;

    public:
        ConstantOfShape(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
         
         //std::vector<uint32_t> output_shape();
   
        ~ConstantOfShape(){}
    };
}


namespace backend {    
    ConstantOfShape::ConstantOfShape(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/constantofshape.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* ConstantOfShape::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }


};

#endif
