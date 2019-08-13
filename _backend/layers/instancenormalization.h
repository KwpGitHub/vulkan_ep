#ifndef INSTANCENORMALIZATION_H
#define INSTANCENORMALIZATION_H //InstanceNormalization

#include "../layer.h"

//INPUTS:                   input_input, scale_input, B_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      epsilon
//OPTIONAL_PARAMETERS_TYPE: float



namespace backend {
    class InstanceNormalization : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            float epsilon;
			
            //input
            Shape_t input_input; Shape_t scale_input; Shape_t B_input;
            
            //output
            Shape_t output_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        InstanceNormalization(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward() { program->run(); }
        
        float epsilon;
		
        //input
        std::string input_input; std::string scale_input; std::string B_input;
        
        //output
        std::string output_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~InstanceNormalization() {}
    };
}


namespace backend {    
    InstanceNormalization::InstanceNormalization(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
        program = new vuh::Program<Specs, Params>(*_get_device(), std::string(file_path + "/shaders/bin/instancenormalization.spv").c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind({epsilon, tensor_dict[input_input]->shape(), tensor_dict[scale_input]->shape(), tensor_dict[B_input]->shape(), tensor_dict[output_output]->shape()} 
                        
                        , tensor_dict[input_input], tensor_dict[scale_input], tensor_dict[B_input], tensor_dict[output_output] );
    }

    vuh::Device* InstanceNormalization::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
};

#endif
