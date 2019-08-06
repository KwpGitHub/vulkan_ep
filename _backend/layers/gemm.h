#ifndef GEMM_H
#define GEMM_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class Gemm : public Layer {
        struct Params{Shape_t A_t; Shape_t B_t; Shape_t C_t; Shape_t Y_t; float alpha_t; float beta_t; int transA_t; int transB_t;
        };
            
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string A; std::string B; std::string C; std::string Y;
        //parameter 
        Shape_t A_t; Shape_t B_t; Shape_t C_t; Shape_t Y_t; float alpha_t; float beta_t; int transA_t; int transB_t;

    public:
        Gemm(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            A = i[0]; B = i[1]; C = i[2];
            Y = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/gemm.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({A_t, B_t, C_t, Y_t, alpha_t, beta_t, transA_t, transB_t }, tensor_dict[Y], tensor_dict[A], tensor_dict[B], tensor_dict[C]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["A"], A_t);
			convert_vec_param(a["B"], B_t);
			convert_vec_param(a["C"], C_t);
			convert_vec_param(a["Y"], Y_t);
			convert_vec_param(a["alpha"], alpha_t);
			convert_vec_param(a["beta"], beta_t);
			convert_vec_param(a["transA"], transA_t);
			convert_vec_param(a["transB"], transB_t);   
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

    
        ~Gemm(){}

    };
}

#endif
