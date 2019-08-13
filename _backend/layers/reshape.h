#ifndef RESHAPE_H
#define RESHAPE_H //Reshape

#include "../layer.h"

//INPUTS:                   data_input, shape_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   reshaped_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 



namespace backend {
    class Reshape : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            
			
            //input
            Shape_t data_input; Shape_t shape_input;
            
            //output
            Shape_t reshaped_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Reshape(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward() { program->run(); }
        
        
		
        //input
        std::string data_input; std::string shape_input;
        
        //output
        std::string reshaped_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~Reshape() {}
    };
}


namespace backend {    
    Reshape::Reshape(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
        program = new vuh::Program<Specs, Params>(*_get_device(), std::string(file_path + "/shaders/bin/reshape.spv").c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind({tensor_dict[data_input]->shape(), tensor_dict[shape_input]->shape(), tensor_dict[reshaped_output]->shape()} 
                        
                        , tensor_dict[data_input], tensor_dict[shape_input], tensor_dict[reshaped_output] );
    }

    vuh::Device* Reshape::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
};

#endif
