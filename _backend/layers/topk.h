#ifndef TOPK_H
#define TOPK_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class TopK : public Layer {
        struct Params{Shape_t X_t; Shape_t K_t; Shape_t Values_t; Shape_t Indices_t; int axis_t;
        };
            
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string X; std::string K; std::string Values; std::string Indices;
        //parameter 
        Shape_t X_t; Shape_t K_t; Shape_t Values_t; Shape_t Indices_t; int axis_t;

    public:
        TopK(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            X = i[0]; K = i[1];
            Values = o[0]; Indices = o[1];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/topk.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({X_t, K_t, Values_t, Indices_t, axis_t }, tensor_dict[Values], tensor_dict[Indices], tensor_dict[X], tensor_dict[K]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["X"], X_t);
			convert_vec_param(a["K"], K_t);
			convert_vec_param(a["Values"], Values_t);
			convert_vec_param(a["Indices"], Indices_t);
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

    
        ~TopK(){}

    };
}

#endif
