#ifndef IF_H
#define IF_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class If : public Layer {
        struct Params{Shape_t cond_t; Shape_t outputs_t; //graph else_branch_t; //graph then_branch_t;
        };
            
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string cond; std::string outputs;
        //parameter 
        Shape_t cond_t; Shape_t outputs_t; //graph else_branch_t; //graph then_branch_t;

    public:
        If(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            cond = i[0];
            outputs = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/if.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({cond_t, outputs_t, else_branch_t, then_branch_t }, tensor_dict[outputs], tensor_dict[cond]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["cond"], cond_t);
			convert_vec_param(a["outputs"], outputs_t);
			convert_vec_param(a["else_branch"], else_branch_t);
			convert_vec_param(a["then_branch"], then_branch_t);   
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

    
        ~If(){}

    };
}

#endif
