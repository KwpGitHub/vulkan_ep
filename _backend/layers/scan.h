#ifndef SCAN_H
#define SCAN_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class Scan : public Layer {
        struct Params{Shape_t initial_state_and_scan_inputs_t; Shape_t final_state_and_scan_outputs_t; //graph body_t; int num_scan_inputs_t; Shape_t scan_input_axes_t; Shape_t scan_input_directions_t; Shape_t scan_output_axes_t; Shape_t scan_output_directions_t;
        };
            
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string initial_state_and_scan_inputs; std::string final_state_and_scan_outputs;
        //parameter 
        Shape_t initial_state_and_scan_inputs_t; Shape_t final_state_and_scan_outputs_t; //graph body_t; int num_scan_inputs_t; Shape_t scan_input_axes_t; Shape_t scan_input_directions_t; Shape_t scan_output_axes_t; Shape_t scan_output_directions_t;

    public:
        Scan(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            initial_state_and_scan_inputs = i[0];
            final_state_and_scan_outputs = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/scan.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({initial_state_and_scan_inputs_t, final_state_and_scan_outputs_t, body_t, num_scan_inputs_t, scan_input_axes_t, scan_input_directions_t, scan_output_axes_t, scan_output_directions_t }, tensor_dict[final_state_and_scan_outputs], tensor_dict[initial_state_and_scan_inputs]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["initial_state_and_scan_inputs"], initial_state_and_scan_inputs_t);
			convert_vec_param(a["final_state_and_scan_outputs"], final_state_and_scan_outputs_t);
			convert_vec_param(a["body"], body_t);
			convert_vec_param(a["num_scan_inputs"], num_scan_inputs_t);
			convert_vec_param(a["scan_input_axes"], scan_input_axes_t);
			convert_vec_param(a["scan_input_directions"], scan_input_directions_t);
			convert_vec_param(a["scan_output_axes"], scan_output_axes_t);
			convert_vec_param(a["scan_output_directions"], scan_output_directions_t);   
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

    
        ~Scan(){}

    };
}

#endif
