#ifndef AND_H
#define AND_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class And : public Layer {
        struct Params{
            
			Shape_t A;
			Shape_t B;
			Shape_t C;
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
		std::string A;
		std::string B;

        //outputs
		std::string C;


    public:
        And(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
        //inputs
			 A = i[0];
			 B = i[1];
        //outputs
			 C = o[0];

            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/and.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({}, tensor_dict[C], tensor_dict[A],tensor_dict[B]);

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

        ~And(){}

    };
}

#endif
