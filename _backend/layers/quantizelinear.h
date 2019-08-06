#ifndef QUANTIZELINEAR_H
#define QUANTIZELINEAR_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class QuantizeLinear : public Layer {
        struct Params{Shape_t x_t; Shape_t y_scale_t; Shape_t y_zero_point_t; Shape_t y_t;
        };
            
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string x; std::string y_scale; std::string y_zero_point; std::string y;
        //parameter 
        Shape_t x_t; Shape_t y_scale_t; Shape_t y_zero_point_t; Shape_t y_t;

    public:
        QuantizeLinear(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            x = i[0]; y_scale = i[1]; y_zero_point = i[2];
            y = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/quantizelinear.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({x_t, y_scale_t, y_zero_point_t, y_t }, tensor_dict[y], tensor_dict[x], tensor_dict[y_scale], tensor_dict[y_zero_point]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["x"], x_t);
			convert_vec_param(a["y_scale"], y_scale_t);
			convert_vec_param(a["y_zero_point"], y_zero_point_t);
			convert_vec_param(a["y"], y_t);   
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

    
        ~QuantizeLinear(){}

    };
}

#endif
