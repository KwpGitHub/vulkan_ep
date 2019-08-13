#ifndef LESS_H
#define LESS_H //Less

#include "../layer.h"

//INPUTS:                   A_input, B_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   C_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 



namespace backend {
    class Less : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            
			
            //input
            Shape_t A_input; Shape_t B_input;
            
            //output
            Shape_t C_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Less(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward() { program->run(); }
        
        
		
        //input
        std::string A_input; std::string B_input;
        
        //output
        std::string C_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~Less() {}
    };
}


namespace backend {    
    Less::Less(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
        program = new vuh::Program<Specs, Params>(*_get_device(), std::string(file_path + "/shaders/bin/less.spv").c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind({tensor_dict[A_input]->shape(), tensor_dict[B_input]->shape(), tensor_dict[C_output]->shape()} 
                        
                        , tensor_dict[A_input], tensor_dict[B_input], tensor_dict[C_output] );
    }

    vuh::Device* Less::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
};

#endif
