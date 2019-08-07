#ifndef REVERSESEQUENCE_H
#define REVERSESEQUENCE_H //ReverseSequence

//INPUTS:                   input, sequence_lens
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      batch_axis, time_axis
//OPTIONAL_PARAMETERS_TYPE: INT, INT

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReverseSequence : public Layer {
        
        vuh::Device* _get_device();

        struct Params{ };
        vuh::Program<Specs, Params>* program;

    public:
        ReverseSequence(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
         
         //std::vector<uint32_t> output_shape();
   
        ~ReverseSequence(){}
    };
}


namespace backend {    
    ReverseSequence::ReverseSequence(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/reversesequence.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* ReverseSequence::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }


};

#endif
