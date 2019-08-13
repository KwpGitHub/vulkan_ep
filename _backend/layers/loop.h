#ifndef LOOP_H
#define LOOP_H //Loop

#include "../layer.h"

//INPUTS:                   
//OPTIONAL_INPUTS:          M_input_o, cond_input_o
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               body
//PARAMETER_TYPES:          int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 



namespace backend {
    class Loop : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int body;
			
            //input
            
            Shape_t M_input_o; Shape_t cond_input_o;
            //output
            
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Loop(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward() { program->run(); }
        
        int body;
		
        //input
        
        std::string M_input_o; std::string cond_input_o;
        //output
        
        
        //std::vector<uint32_t> output_shape();
   
        ~Loop() {}
    };
}


namespace backend {    
    Loop::Loop(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
        program = new vuh::Program<Specs, Params>(*_get_device(), std::string(file_path + "/shaders/bin/loop.spv").c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind({body, tensor_dict[M_input_o]->shape(), tensor_dict[cond_input_o]->shape()} 
                        
                        , tensor_dict[M_input_o], tensor_dict[cond_input_o] );
    }

    vuh::Device* Loop::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
};

#endif
