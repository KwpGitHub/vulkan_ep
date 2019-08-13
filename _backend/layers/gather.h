#ifndef GATHER_H
#define GATHER_H //Gather

#include "../layer.h"

//INPUTS:                   data_input, indices_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis
//OPTIONAL_PARAMETERS_TYPE: int



namespace backend {
    class Gather : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int axis;
			
            //input
            Shape_t data_input; Shape_t indices_input;
            
            //output
            Shape_t output_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Gather(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward() { program->run(); }
        
        int axis;
		
        //input
        std::string data_input; std::string indices_input;
        
        //output
        std::string output_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~Gather() {}
    };
}


namespace backend {    
    Gather::Gather(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
        program = new vuh::Program<Specs, Params>(*_get_device(), std::string(file_path + "/shaders/bin/gather.spv").c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind({axis, tensor_dict[data_input]->shape(), tensor_dict[indices_input]->shape(), tensor_dict[output_output]->shape()} 
                        
                        , tensor_dict[data_input], tensor_dict[indices_input], tensor_dict[output_output] );
    }

    vuh::Device* Gather::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
};

#endif
