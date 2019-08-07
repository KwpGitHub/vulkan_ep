#ifndef RNN_H
#define RNN_H //RNN

//INPUTS:                   X, W, R
//OPTIONAL_INPUTS:          B, sequence_lens, initial_h
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         Y, Y_h
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      activation_alpha, activation_beta, activations, clip, direction, hidden_size
//OPTIONAL_PARAMETERS_TYPE: FLOATS, FLOATS, STRINGS, FLOAT, STRING, INT

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class RNN : public Layer {
        
        vuh::Device* _get_device();

        struct Params{ };
        vuh::Program<Specs, Params>* program;

    public:
        RNN(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
         
         //std::vector<uint32_t> output_shape();
   
        ~RNN(){}
    };
}


namespace backend {    
    RNN::RNN(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/rnn.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* RNN::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }


};

#endif
