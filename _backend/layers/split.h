#ifndef SPLIT_H
#define SPLIT_H //Split

#include "../layer.h"

//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis, split
//OPTIONAL_PARAMETERS_TYPE: int, Shape_t



namespace backend {
    class Split : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int axis; Shape_t split;
			
            //input
            Shape_t input_input;
            
            //output
            
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Split(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward() { program->run(); }
        
        int axis; Shape_t split;
		
        //input
        std::string input_input;
        
        //output
        
        
        //std::vector<uint32_t> output_shape();
   
        ~Split() {}
    };
}


namespace backend {    
    Split::Split(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
        program = new vuh::Program<Specs, Params>(*_get_device(), std::string(file_path + "/shaders/bin/split.spv").c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind({axis, split, tensor_dict[input_input]->shape()} 
                        
                        , tensor_dict[input_input] );
    }

    vuh::Device* Split::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
};

#endif
