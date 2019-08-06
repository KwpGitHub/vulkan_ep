#ifndef CONV_H
#define CONV_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class Conv : public Layer {
        struct Params{Shape_t X_t; Shape_t W_t; Shape_t B_t; Shape_t Y_t; int auto_pad_t; Shape_t dilations_t; int group_t; Shape_t kernel_shape_t; Shape_t pads_t; Shape_t strides_t;
        };
            
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string X; std::string W; std::string B; std::string Y;
        //parameter 
        Shape_t X_t; Shape_t W_t; Shape_t B_t; Shape_t Y_t; int auto_pad_t; Shape_t dilations_t; int group_t; Shape_t kernel_shape_t; Shape_t pads_t; Shape_t strides_t;

    public:
        Conv(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            X = i[0]; W = i[1]; B = i[2];
            Y = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/conv.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({X_t, W_t, B_t, Y_t, auto_pad_t, dilations_t, group_t, kernel_shape_t, pads_t, strides_t }, tensor_dict[Y], tensor_dict[X], tensor_dict[W], tensor_dict[B]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["X"], X_t);
			convert_vec_param(a["W"], W_t);
			convert_vec_param(a["B"], B_t);
			convert_vec_param(a["Y"], Y_t);
			convert_vec_param(a["auto_pad"], auto_pad_t);
			convert_vec_param(a["dilations"], dilations_t);
			convert_vec_param(a["group"], group_t);
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

    
        ~Conv(){}

    };
}

#endif
