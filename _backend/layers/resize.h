#ifndef RESIZE_H
#define RESIZE_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class Resize : public Layer {
        struct Params{Shape_t X_t; Shape_t scales_t; Shape_t Y_t; int mode_t;
        };
            
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string X; std::string scales; std::string Y;
        //parameter 
        Shape_t X_t; Shape_t scales_t; Shape_t Y_t; int mode_t;

    public:
        Resize(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            X = i[0]; scales = i[1];
            Y = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/resize.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({X_t, scales_t, Y_t, mode_t }, tensor_dict[Y], tensor_dict[X], tensor_dict[scales]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["X"], X_t);
			convert_vec_param(a["scales"], scales_t);
			convert_vec_param(a["Y"], Y_t);
			convert_vec_param(a["mode"], mode_t);   
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

    
        ~Resize(){}

    };
}

#endif