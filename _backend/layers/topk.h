#ifndef TOPK_H
#define TOPK_H //TopK

#include "../layer.h"

//INPUTS:                   X_input, K_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Values_output, Indices_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis
//OPTIONAL_PARAMETERS_TYPE: int



namespace backend {
    class TopK : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int axis;
			
            //input
            Shape_t X_input; Shape_t K_input;
            
            //output
            Shape_t Values_output; Shape_t Indices_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        TopK(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward() { program->run(); }
        
        int axis;
		
        //input
        std::string X_input; std::string K_input;
        
        //output
        std::string Values_output; std::string Indices_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~TopK() {}
    };
}


namespace backend {    
    TopK::TopK(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
        program = new vuh::Program<Specs, Params>(*_get_device(), std::string(file_path + "/shaders/bin/topk.spv").c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind({axis, tensor_dict[X_input]->shape(), tensor_dict[K_input]->shape(), tensor_dict[Values_output]->shape(), tensor_dict[Indices_output]->shape()} 
                        
                        , tensor_dict[X_input], tensor_dict[K_input], tensor_dict[Values_output], tensor_dict[Indices_output] );
    }

    vuh::Device* TopK::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
};

#endif
