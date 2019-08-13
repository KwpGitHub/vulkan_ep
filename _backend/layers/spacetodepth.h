#ifndef SPACETODEPTH_H
#define SPACETODEPTH_H //SpaceToDepth

#include "../layer.h"

//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               blocksize
//PARAMETER_TYPES:          int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 



namespace backend {
    class SpaceToDepth : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int blocksize;
			
            //input
            Shape_t input_input;
            
            //output
            Shape_t output_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        SpaceToDepth(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward() { program->run(); }
        
        int blocksize;
		
        //input
        std::string input_input;
        
        //output
        std::string output_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~SpaceToDepth() {}
    };
}


namespace backend {    
    SpaceToDepth::SpaceToDepth(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
        program = new vuh::Program<Specs, Params>(*_get_device(), std::string(file_path + "/shaders/bin/spacetodepth.spv").c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind({blocksize, tensor_dict[input_input]->shape(), tensor_dict[output_output]->shape()} 
                        
                        , tensor_dict[input_input], tensor_dict[output_output] );
    }

    vuh::Device* SpaceToDepth::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
};

#endif
