#ifndef MATMULINTEGER_H
#define MATMULINTEGER_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class MatMulInteger : public Layer {
        struct Params{Shape_t A_t; Shape_t B_t; Shape_t a_zero_point_t; Shape_t b_zero_point_t; Shape_t Y_t;
        };
            
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string A; std::string B; std::string a_zero_point; std::string b_zero_point; std::string Y;
        //parameter 
        Shape_t A_t; Shape_t B_t; Shape_t a_zero_point_t; Shape_t b_zero_point_t; Shape_t Y_t;

    public:
        MatMulInteger(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            A = i[0]; B = i[1]; a_zero_point = i[2]; b_zero_point = i[3];
            Y = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/matmulinteger.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({A_t, B_t, a_zero_point_t, b_zero_point_t, Y_t }, tensor_dict[Y], tensor_dict[A], tensor_dict[B], tensor_dict[a_zero_point], tensor_dict[b_zero_point]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["A"], A_t);
			convert_vec_param(a["B"], B_t);
			convert_vec_param(a["a_zero_point"], a_zero_point_t);
			convert_vec_param(a["b_zero_point"], b_zero_point_t);
			convert_vec_param(a["Y"], Y_t);   
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

    
        ~MatMulInteger(){}

    };
}

#endif
