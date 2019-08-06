#ifndef BATCHNORMALIZATION_H
#define BATCHNORMALIZATION_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class BatchNormalization : public Layer {
        struct Params{
            
			Shape_t X;
			Shape_t scale;
			Shape_t B;
			Shape_t mean;
			Shape_t var;
			Shape_t Y;
			Shape_t mean;
			Shape_t var;
			Shape_t saved_mean;
			Shape_t saved_var;
			float epsilon;
			float momentum;
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
		std::string X;
		std::string scale;
		std::string B;
		std::string mean;
		std::string var;

        //outputs
		std::string Y;
		std::string mean;
		std::string var;
		std::string saved_mean;
		std::string saved_var;


    public:
        BatchNormalization(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
        //inputs
			 X = i[0];
			 scale = i[1];
			 B = i[2];
			 mean = i[3];
			 var = i[4];
        //outputs
			 Y = o[0];
			 mean = o[1];
			 var = o[2];
			 saved_mean = o[3];
			 saved_var = o[4];

            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/batchnormalization.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({}, tensor_dict[Y],tensor_dict[mean],tensor_dict[var],tensor_dict[saved_mean],tensor_dict[saved_var], tensor_dict[X],tensor_dict[scale],tensor_dict[B],tensor_dict[mean],tensor_dict[var]);

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

        ~BatchNormalization(){}

    };
}

#endif
