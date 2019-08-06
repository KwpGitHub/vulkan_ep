#ifndef ONEHOT_H
#define ONEHOT_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class OneHot : public Layer {
        struct Params{Shape_t indices_t; Shape_t depth_t; Shape_t values_t; Shape_t output_t; int axis_t;
        };
            
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string indices; std::string depth; std::string values; std::string output;
        //parameter 
        Shape_t indices_t; Shape_t depth_t; Shape_t values_t; Shape_t output_t; int axis_t;

    public:
        OneHot(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            indices = i[0]; depth = i[1]; values = i[2];
            output = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/onehot.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({indices_t, depth_t, values_t, output_t, axis_t }, tensor_dict[output], tensor_dict[indices], tensor_dict[depth], tensor_dict[values]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["indices"], indices_t);
			convert_vec_param(a["depth"], depth_t);
			convert_vec_param(a["values"], values_t);
			convert_vec_param(a["output"], output_t);
			convert_vec_param(a["axis"], axis_t);   
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

    
        ~OneHot(){}

    };
}

#endif
