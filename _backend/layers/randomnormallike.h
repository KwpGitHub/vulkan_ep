#ifndef RANDOMNORMALLIKE_H
#define RANDOMNORMALLIKE_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class RandomNormalLike : public Layer {
        struct Params{Shape_t input_t; Shape_t output_t; int dtype_t; float mean_t; float scale_t; float seed_t;
        };
            
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string input; std::string output;
        //parameter 
        Shape_t input_t; Shape_t output_t; int dtype_t; float mean_t; float scale_t; float seed_t;

    public:
        RandomNormalLike(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            input = i[0];
            output = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/randomnormallike.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({input_t, output_t, dtype_t, mean_t, scale_t, seed_t }, tensor_dict[output], tensor_dict[input]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["input"], input_t);
			convert_vec_param(a["output"], output_t);
			convert_vec_param(a["dtype"], dtype_t);
			convert_vec_param(a["mean"], mean_t);
			convert_vec_param(a["scale"], scale_t);
			convert_vec_param(a["seed"], seed_t);   
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

    
        ~RandomNormalLike(){}

    };
}

#endif
