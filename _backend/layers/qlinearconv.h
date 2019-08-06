#ifndef QLINEARCONV_H
#define QLINEARCONV_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class QLinearConv : public Layer {
        struct Params{Shape_t x_t; Shape_t x_scale_t; Shape_t x_zero_point_t; Shape_t w_t; Shape_t w_scale_t; Shape_t w_zero_point_t; Shape_t y_scale_t; Shape_t y_zero_point_t; Shape_t B_t; Shape_t y_t; int auto_pad_t; Shape_t dilations_t; int group_t; Shape_t kernel_shape_t; Shape_t pads_t; Shape_t strides_t;
        };
            
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string x; std::string x_scale; std::string x_zero_point; std::string w; std::string w_scale; std::string w_zero_point; std::string y_scale; std::string y_zero_point; std::string B; std::string y;
        //parameter 
        Shape_t x_t; Shape_t x_scale_t; Shape_t x_zero_point_t; Shape_t w_t; Shape_t w_scale_t; Shape_t w_zero_point_t; Shape_t y_scale_t; Shape_t y_zero_point_t; Shape_t B_t; Shape_t y_t; int auto_pad_t; Shape_t dilations_t; int group_t; Shape_t kernel_shape_t; Shape_t pads_t; Shape_t strides_t;

    public:
        QLinearConv(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            x = i[0]; x_scale = i[1]; x_zero_point = i[2]; w = i[3]; w_scale = i[4]; w_zero_point = i[5]; y_scale = i[6]; y_zero_point = i[7]; B = i[8];
            y = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/qlinearconv.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({x_t, x_scale_t, x_zero_point_t, w_t, w_scale_t, w_zero_point_t, y_scale_t, y_zero_point_t, B_t, y_t, auto_pad_t, dilations_t, group_t, kernel_shape_t, pads_t, strides_t }, tensor_dict[y], tensor_dict[x], tensor_dict[x_scale], tensor_dict[x_zero_point], tensor_dict[w], tensor_dict[w_scale], tensor_dict[w_zero_point], tensor_dict[y_scale], tensor_dict[y_zero_point], tensor_dict[B]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["x"], x_t);
			convert_vec_param(a["x_scale"], x_scale_t);
			convert_vec_param(a["x_zero_point"], x_zero_point_t);
			convert_vec_param(a["w"], w_t);
			convert_vec_param(a["w_scale"], w_scale_t);
			convert_vec_param(a["w_zero_point"], w_zero_point_t);
			convert_vec_param(a["y_scale"], y_scale_t);
			convert_vec_param(a["y_zero_point"], y_zero_point_t);
			convert_vec_param(a["B"], B_t);
			convert_vec_param(a["y"], y_t);
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

    
        ~QLinearConv(){}

    };
}

#endif
