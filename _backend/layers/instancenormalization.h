#ifndef INSTANCENORMALIZATION_H
#define INSTANCENORMALIZATION_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class InstanceNormalization : public Layer {
        struct Params{Shape_t input_t; Shape_t scale_t; Shape_t B_t; Shape_t output_t; float epsilon_t;
        };
            
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string input; std::string scale; std::string B; std::string output;
        //parameter 
        Shape_t input_t; Shape_t scale_t; Shape_t B_t; Shape_t output_t; float epsilon_t;

    public:
        InstanceNormalization(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            input = i[0]; scale = i[1]; B = i[2];
            output = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/instancenormalization.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({input_t, scale_t, B_t, output_t, epsilon_t }, tensor_dict[output], tensor_dict[input], tensor_dict[scale], tensor_dict[B]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["input"], input_t);
			convert_vec_param(a["scale"], scale_t);
			convert_vec_param(a["B"], B_t);
			convert_vec_param(a["output"], output_t);
			convert_vec_param(a["epsilon"], epsilon_t);   
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

    
        ~InstanceNormalization(){}

    };
}

#endif
