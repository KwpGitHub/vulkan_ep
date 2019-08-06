#ifndef LINEARCLASSIFIER_H
#define LINEARCLASSIFIER_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class LinearClassifier : public Layer {
        struct Params{Shape_t X_t; Shape_t Y_t; Shape_t Z_t; Shape_t classlabels_ints_t; int* classlabels_strings_t; int multi_class_t; int post_transform_t;
        };
            
		Tensor* coefficients;
		Tensor* intercepts;
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string X; std::string Y; std::string Z;
        //parameter 
        Shape_t X_t; Shape_t Y_t; Shape_t Z_t; Shape_t classlabels_ints_t; int* classlabels_strings_t; int multi_class_t; int post_transform_t;

    public:
        LinearClassifier(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            X = i[0];
            Y = o[0]; Z = o[1];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/linearclassifier.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({X_t, Y_t, Z_t, classlabels_ints_t, classlabels_strings_t, multi_class_t, post_transform_t }, tensor_dict[Y], tensor_dict[Z], tensor_dict[X]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["X"], X_t);
			convert_vec_param(a["Y"], Y_t);
			convert_vec_param(a["Z"], Z_t);
			convert_vec_param(a["classlabels_ints"], classlabels_ints_t);
			convert_vec_param(a["classlabels_strings"], classlabels_strings_t);
			convert_vec_param(a["multi_class"], multi_class_t);
			convert_vec_param(a["post_transform"], post_transform_t);   
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

    
        ~LinearClassifier(){}

    };
}

#endif
