#ifndef GRU_H
#define GRU_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class GRU : public Layer {
        struct Params{Shape_t X_t; Shape_t W_t; Shape_t R_t; Shape_t B_t; Shape_t sequence_lens_t; Shape_t initial_h_t; Shape_t Y_t; Shape_t Y_h_t; int* activations_t; float clip_t; int direction_t; int hidden_size_t; int linear_before_reset_t;
        };
            
		Tensor* activation_alpha;
		Tensor* activation_beta;
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string X; std::string W; std::string R; std::string B; std::string sequence_lens; std::string initial_h; std::string Y; std::string Y_h;
        //parameter 
        Shape_t X_t; Shape_t W_t; Shape_t R_t; Shape_t B_t; Shape_t sequence_lens_t; Shape_t initial_h_t; Shape_t Y_t; Shape_t Y_h_t; int* activations_t; float clip_t; int direction_t; int hidden_size_t; int linear_before_reset_t;

    public:
        GRU(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            X = i[0]; W = i[1]; R = i[2]; B = i[3]; sequence_lens = i[4]; initial_h = i[5];
            Y = o[0]; Y_h = o[1];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/gru.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({X_t, W_t, R_t, B_t, sequence_lens_t, initial_h_t, Y_t, Y_h_t, activations_t, clip_t, direction_t, hidden_size_t, linear_before_reset_t }, tensor_dict[Y], tensor_dict[Y_h], tensor_dict[X], tensor_dict[W], tensor_dict[R], tensor_dict[B], tensor_dict[sequence_lens], tensor_dict[initial_h]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["X"], X_t);
			convert_vec_param(a["W"], W_t);
			convert_vec_param(a["R"], R_t);
			convert_vec_param(a["B"], B_t);
			convert_vec_param(a["sequence_lens"], sequence_lens_t);
			convert_vec_param(a["initial_h"], initial_h_t);
			convert_vec_param(a["Y"], Y_t);
			convert_vec_param(a["Y_h"], Y_h_t);
			convert_vec_param(a["activations"], activations_t);
			convert_vec_param(a["clip"], clip_t);
			convert_vec_param(a["direction"], direction_t);
			convert_vec_param(a["hidden_size"], hidden_size_t);
			convert_vec_param(a["linear_before_reset"], linear_before_reset_t);   
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

    
        ~GRU(){}

    };
}

#endif
