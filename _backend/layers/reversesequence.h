#ifndef REVERSESEQUENCE_H
#define REVERSESEQUENCE_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class ReverseSequence : public Layer {
        struct Params{Shape_t input_t; Shape_t sequence_lens_t; Shape_t Y_t; int batch_axis_t; int time_axis_t;
        };
            
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string input; std::string sequence_lens; std::string Y;
        //parameter 
        Shape_t input_t; Shape_t sequence_lens_t; Shape_t Y_t; int batch_axis_t; int time_axis_t;

    public:
        ReverseSequence(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            input = i[0]; sequence_lens = i[1];
            Y = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/reversesequence.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({input_t, sequence_lens_t, Y_t, batch_axis_t, time_axis_t }, tensor_dict[Y], tensor_dict[input], tensor_dict[sequence_lens]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["input"], input_t);
			convert_vec_param(a["sequence_lens"], sequence_lens_t);
			convert_vec_param(a["Y"], Y_t);
			convert_vec_param(a["batch_axis"], batch_axis_t);
			convert_vec_param(a["time_axis"], time_axis_t);   
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

    
        ~ReverseSequence(){}

    };
}

#endif
