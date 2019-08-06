#ifndef MAX_H
#define MAX_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class Max : public Layer {
        struct Params{Shape_t data_0_t; Shape_t max_t;
        };
            
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string data_0; std::string max;
        //parameter 
        Shape_t data_0_t; Shape_t max_t;

    public:
        Max(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            data_0 = i[0];
            max = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/max.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({data_0_t, max_t }, tensor_dict[max], tensor_dict[data_0]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["data_0"], data_0_t);
			convert_vec_param(a["max"], max_t);   
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

    
        ~Max(){}

    };
}

#endif
