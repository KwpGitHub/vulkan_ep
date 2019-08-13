#ifndef ONEHOTENCODER_H
#define ONEHOTENCODER_H //OneHotEncoder

#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      cats_int64s, cats_strings, zeros
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*, int



namespace backend {
    class OneHotEncoder : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            Shape_t cats_int64s; int zeros;
			Shape_t cats_strings;
            //input
            Shape_t X_input;
            
            //output
            Shape_t Y_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        OneHotEncoder(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward() { program->run(); }
        
        Shape_t cats_int64s; Tensor* cats_strings; int zeros;
		Shape_t cats_strings_s;
        //input
        std::string X_input;
        
        //output
        std::string Y_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~OneHotEncoder() {}
    };
}


namespace backend {    
    OneHotEncoder::OneHotEncoder(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
        program = new vuh::Program<Specs, Params>(*_get_device(), std::string(file_path + "/shaders/bin/onehotencoder.spv").c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind({cats_int64s, zeros, cats_strings_s, tensor_dict[X_input]->shape(), tensor_dict[Y_output]->shape()} 
                        , *cats_strings
                        , tensor_dict[X_input], tensor_dict[Y_output] );
    }

    vuh::Device* OneHotEncoder::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
};

#endif
