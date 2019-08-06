#ifndef LOOP_H
#define LOOP_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class Loop : public Layer {
        struct Params{Shape_t M_t; Shape_t cond_t; Shape_t v_initial_t; Shape_t v_final_and_scan_outputs_t; //graph body_t;
        };
            
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string M; std::string cond; std::string v_initial; std::string v_final_and_scan_outputs;
        //parameter 
        Shape_t M_t; Shape_t cond_t; Shape_t v_initial_t; Shape_t v_final_and_scan_outputs_t; //graph body_t;

    public:
        Loop(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            M = i[0]; cond = i[1]; v_initial = i[2];
            v_final_and_scan_outputs = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/loop.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({M_t, cond_t, v_initial_t, v_final_and_scan_outputs_t, body_t }, tensor_dict[v_final_and_scan_outputs], tensor_dict[M], tensor_dict[cond], tensor_dict[v_initial]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["M"], M_t);
			convert_vec_param(a["cond"], cond_t);
			convert_vec_param(a["v_initial"], v_initial_t);
			convert_vec_param(a["v_final_and_scan_outputs"], v_final_and_scan_outputs_t);
			convert_vec_param(a["body"], body_t);   
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

    
        ~Loop(){}

    };
}

#endif
