#ifndef SCAN_H
#define SCAN_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class Scan : public Layer {
        struct Params{
            
			Shape_t initial_state_and_scan_inputs;
			Shape_t final_state_and_scan_outputs;
			//graph body;
			int num_scan_inputs;
			int* scan_input_axes;
			int* scan_input_directions;
			int* scan_output_axes;
			int* scan_output_directions;
        };

        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) {
                    return tensor_dict[t_name]->dev;
                }
            }
            return device;
        }

        //inputs
		std::string initial_state_and_scan_inputs;

        //outputs
		std::string final_state_and_scan_outputs;


    public:
        Scan(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
        //inputs
			 initial_state_and_scan_inputs = i[0];
        //outputs
			 final_state_and_scan_outputs = o[0];

            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/scan.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({}, tensor_dict[final_state_and_scan_outputs], tensor_dict[initial_state_and_scan_inputs]);

        }
        
        //vuh::Array<float>& operator()(const vuh::Array<float>& t) {            
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

        void build_pipeline(){
           // std::vector<Tensor> x;
           // for(auto t_name : inputs)
           //     x.push_back(*tensor_dict[t_name]);
            //program->bind({}, );
		    
        }

        ~Scan(){}

    };
}

#endif
