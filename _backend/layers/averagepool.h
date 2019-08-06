#ifndef AVERAGEPOOL_H
#define AVERAGEPOOL_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class AveragePool : public Layer {
        struct Params{Shape_t X_t; Shape_t Y_t; int auto_pad_t; int ceil_mode_t; int count_include_pad_t; Shape_t kernel_shape_t; Shape_t pads_t; Shape_t strides_t;
        };
            
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
        Shape_t X_t; Shape_t Y_t; int auto_pad_t; int ceil_mode_t; int count_include_pad_t; Shape_t kernel_shape_t; Shape_t pads_t; Shape_t strides_t;

    public:
        AveragePool(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            X = i[0];
            Y = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/averagepool.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({X_t, Y_t, auto_pad_t, ceil_mode_t, count_include_pad_t, kernel_shape_t, pads_t, strides_t }, tensor_dict[Y], tensor_dict[X]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["X"], X_t);
			convert_vec_param(a["Y"], Y_t);
			convert_vec_param(a["auto_pad"], auto_pad_t);
			convert_vec_param(a["ceil_mode"], ceil_mode_t);
			convert_vec_param(a["count_include_pad"], count_include_pad_t);
			convert_vec_param(a["kernel_shape"], kernel_shape_t);
			convert_vec_param(a["pads"], pads_t);
			convert_vec_param(a["strides"], strides_t);   
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

    
        ~AveragePool(){}

    };
}

#endif
