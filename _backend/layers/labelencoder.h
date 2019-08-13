#ifndef LABELENCODER_H
#define LABELENCODER_H //LabelEncoder

#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      default_float, default_int64, default_string, keys_floats, keys_int64s, keys_strings, values_floats, values_int64s, values_strings
//OPTIONAL_PARAMETERS_TYPE: float, int, int, Tensor*, Shape_t, Tensor*, Tensor*, Shape_t, Tensor*



namespace backend {
    class LabelEncoder : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            float default_float; int default_int64; int default_string; Shape_t keys_int64s; Shape_t values_int64s;
			Shape_t keys_floats; Shape_t keys_strings; Shape_t values_floats; Shape_t values_strings;
            //input
            Shape_t X_input;
            
            //output
            Shape_t Y_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        LabelEncoder(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward() { program->run(); }
        
        float default_float; int default_int64; int default_string; Tensor* keys_floats; Shape_t keys_int64s; Tensor* keys_strings; Tensor* values_floats; Shape_t values_int64s; Tensor* values_strings;
		Shape_t keys_floats_s; Shape_t keys_strings_s; Shape_t values_floats_s; Shape_t values_strings_s;
        //input
        std::string X_input;
        
        //output
        std::string Y_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~LabelEncoder() {}
    };
}


namespace backend {    
    LabelEncoder::LabelEncoder(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
        program = new vuh::Program<Specs, Params>(*_get_device(), std::string(file_path + "/shaders/bin/labelencoder.spv").c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind({default_float, default_int64, default_string, keys_int64s, values_int64s, keys_floats_s, keys_strings_s, values_floats_s, values_strings_s, tensor_dict[X_input]->shape(), tensor_dict[Y_output]->shape()} 
                        , *keys_floats, *keys_strings, *values_floats, *values_strings
                        , tensor_dict[X_input], tensor_dict[Y_output] );
    }

    vuh::Device* LabelEncoder::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
};

#endif
