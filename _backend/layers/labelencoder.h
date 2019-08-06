#ifndef LABELENCODER_H
#define LABELENCODER_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class LabelEncoder : public Layer {
        struct Params{Shape_t X_t; Shape_t Y_t; float default_float_t; int default_int64_t; int default_string_t; Shape_t keys_int64s_t; int* keys_strings_t; Shape_t values_int64s_t; int* values_strings_t;
        };
            
		Tensor* keys_floats;
		Tensor* values_floats;
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string X; std::string Y;
        //parameter 
        Shape_t X_t; Shape_t Y_t; float default_float_t; int default_int64_t; int default_string_t; Shape_t keys_int64s_t; int* keys_strings_t; Shape_t values_int64s_t; int* values_strings_t;

    public:
        LabelEncoder(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            X = i[0];
            Y = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/labelencoder.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({X_t, Y_t, default_float_t, default_int64_t, default_string_t, keys_int64s_t, keys_strings_t, values_int64s_t, values_strings_t }, tensor_dict[Y], tensor_dict[X]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["X"], X_t);
			convert_vec_param(a["Y"], Y_t);
			convert_vec_param(a["default_float"], default_float_t);
			convert_vec_param(a["default_int64"], default_int64_t);
			convert_vec_param(a["default_string"], default_string_t);
			convert_vec_param(a["keys_int64s"], keys_int64s_t);
			convert_vec_param(a["keys_strings"], keys_strings_t);
			convert_vec_param(a["values_int64s"], values_int64s_t);
			convert_vec_param(a["values_strings"], values_strings_t);   
        }

        //Tensor* operator()(const Tensor* t) {            
        //}

		void forward(){
		}

       /* std::vector<uint32_t> output_shape(){
            for(auto t_name : inputs){
                if(tensor_dict.end() == tensor_dict.find(t_name) && layer_dict.end() != layer_dict.find(t_name)){
                    //need to do math
                    return layer_dict[t_name]->output_shape();
                }
                else if (tensor_dict.end() != tensor_dict.find(t_name) && layer_dict.end() == layer_dict.find(t_name)){
                    //need to do math
                    return tensor_dict[t_name]->dims;
                }

            }
            for(auto t_name : outputs){
                if(tensor_dict.end() != tensor_dict.find(t_name) && layer_dict.end() == layer_dict.find(t_name)){
                    return tensor_dict[t_name]->dims;
                }
            }
        }*/

    
        ~LabelEncoder(){}

    };
}

#endif
