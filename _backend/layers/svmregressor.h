#ifndef SVMREGRESSOR_H
#define SVMREGRESSOR_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class SVMRegressor : public Layer {
        struct Params{Shape_t X_t; Shape_t Y_t; int kernel_type_t; int n_supports_t; int one_class_t; int post_transform_t;
        };
            
		Tensor* coefficients;
		Tensor* kernel_params;
		Tensor* rho;
		Tensor* support_vectors;
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
        Shape_t X_t; Shape_t Y_t; int kernel_type_t; int n_supports_t; int one_class_t; int post_transform_t;

    public:
        SVMRegressor(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            X = i[0];
            Y = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/svmregressor.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({X_t, Y_t, kernel_type_t, n_supports_t, one_class_t, post_transform_t }, tensor_dict[Y], tensor_dict[X]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["X"], X_t);
			convert_vec_param(a["Y"], Y_t);
			convert_vec_param(a["kernel_type"], kernel_type_t);
			convert_vec_param(a["n_supports"], n_supports_t);
			convert_vec_param(a["one_class"], one_class_t);
			convert_vec_param(a["post_transform"], post_transform_t);   
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

    
        ~SVMRegressor(){}

    };
}

#endif
