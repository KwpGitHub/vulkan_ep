#ifndef SLICE_H
#define SLICE_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class Slice : public Layer {
        struct Params{Shape_t data_t; Shape_t starts_t; Shape_t ends_t; Shape_t axes_t; Shape_t steps_t; Shape_t output_t;
        };
            
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string data; std::string starts; std::string ends; std::string axes; std::string steps; std::string output;
        //parameter 
        Shape_t data_t; Shape_t starts_t; Shape_t ends_t; Shape_t axes_t; Shape_t steps_t; Shape_t output_t;

    public:
        Slice(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            data = i[0]; starts = i[1]; ends = i[2]; axes = i[3]; steps = i[4];
            output = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/slice.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({data_t, starts_t, ends_t, axes_t, steps_t, output_t }, tensor_dict[output], tensor_dict[data], tensor_dict[starts], tensor_dict[ends], tensor_dict[axes], tensor_dict[steps]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["data"], data_t);
			convert_vec_param(a["starts"], starts_t);
			convert_vec_param(a["ends"], ends_t);
			convert_vec_param(a["axes"], axes_t);
			convert_vec_param(a["steps"], steps_t);
			convert_vec_param(a["output"], output_t);   
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

    
        ~Slice(){}

    };
}

#endif
