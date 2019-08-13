#ifndef REDUCEPROD_H
#define REDUCEPROD_H //ReduceProd

#include "../layer.h"

//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   reduced_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axes, keepdims
//OPTIONAL_PARAMETERS_TYPE: Shape_t, int



namespace backend {
    class ReduceProd : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            Shape_t axes; int keepdims;
			
            //input
            Shape_t data_input;
            
            //output
            Shape_t reduced_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        ReduceProd(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward() { program->run(); }
        
        Shape_t axes; int keepdims;
		
        //input
        std::string data_input;
        
        //output
        std::string reduced_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~ReduceProd() {}
    };
}


namespace backend {    
    ReduceProd::ReduceProd(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
        program = new vuh::Program<Specs, Params>(*_get_device(), std::string(file_path + "/shaders/bin/reduceprod.spv").c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind({axes, keepdims, tensor_dict[data_input]->shape(), tensor_dict[reduced_output]->shape()} 
                        
                        , tensor_dict[data_input], tensor_dict[reduced_output] );
    }

    vuh::Device* ReduceProd::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
};

#endif
