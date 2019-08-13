#ifndef POW_H
#define POW_H //Pow

#include "../layer.h"

//INPUTS:                   X_input, Y_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Z_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 



namespace backend {
    class Pow : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            
			
            //input
            Shape_t X_input; Shape_t Y_input;
            
            //output
            Shape_t Z_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        Pow(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward() { program->run(); }
        
        
		
        //input
        std::string X_input; std::string Y_input;
        
        //output
        std::string Z_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~Pow() {}
    };
}


namespace backend {    
    Pow::Pow(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
        program = new vuh::Program<Specs, Params>(*_get_device(), std::string(file_path + "/shaders/bin/pow.spv").c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind({tensor_dict[X_input]->shape(), tensor_dict[Y_input]->shape(), tensor_dict[Z_output]->shape()} 
                        
                        , tensor_dict[X_input], tensor_dict[Y_input], tensor_dict[Z_output] );
    }

    vuh::Device* Pow::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
};

#endif
