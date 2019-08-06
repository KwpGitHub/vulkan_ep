#ifndef TREEENSEMBLECLASSIFIER_H
#define TREEENSEMBLECLASSIFIER_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class TreeEnsembleClassifier : public Layer {
        struct Params{
            
			Shape_t X;
			Shape_t Y;
			Shape_t Z;
			float* base_values;
			int* class_ids;
			int* class_nodeids;
			int* class_treeids;
			float* class_weights;
			int* classlabels_int64s;
			std::vector<std::string> classlabels_strings;
			int* nodes_falsenodeids;
			int* nodes_featureids;
			float* nodes_hitrates;
			int* nodes_missing_value_tracks_true;
			std::vector<std::string> nodes_modes;
			int* nodes_nodeids;
			int* nodes_treeids;
			int* nodes_truenodeids;
			float* nodes_values;
			int post_transform;
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

        //outputs
		std::string Y;
		std::string Z;


    public:
        TreeEnsembleClassifier(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
        //inputs
			 X = i[0];
        //outputs
			 Y = o[0];
			 Z = o[1];

            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/treeensembleclassifier.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({}, tensor_dict[Y],tensor_dict[Z], tensor_dict[X]);

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

        ~TreeEnsembleClassifier(){}

    };
}

#endif
