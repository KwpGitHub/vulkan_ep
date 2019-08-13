#ifndef WHERE_H
#define WHERE_H //Where

#include "../layer.h"

//INPUTS:                   condition_input, X_input, Y_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 



namespace backend {
    class Where : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            
			
            //input
            Shape_t condition_input; Shape_t X_input; Shape_t Y_input;
            
            //output
            Shape_t output_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Where(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward() { program->run(); }
        
        
		
        //input
        std::string condition_input; std::string X_input; std::string Y_input;
        
        //output
        std::string output_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~Where() {}
    };
}


namespace backend {    
    Where::Where(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
        program = new vuh::Program<Specs, Params>(*_get_device(), std::string(file_path + "/shaders/bin/where.spv").c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind({tensor_dict[condition_input]->shape(), tensor_dict[X_input]->shape(), tensor_dict[Y_input]->shape(), tensor_dict[output_output]->shape()} 
                        
                        , tensor_dict[condition_input], tensor_dict[X_input], tensor_dict[Y_input], tensor_dict[output_output] );
    }

    vuh::Device* Where::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
};

#endif
