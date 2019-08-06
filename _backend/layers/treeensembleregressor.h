#ifndef TREEENSEMBLEREGRESSOR_H
#define TREEENSEMBLEREGRESSOR_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class TreeEnsembleRegressor : public Layer {
        struct Params{Shape_t X_t; Shape_t Y_t; int aggregate_function_t; int n_targets_t; Shape_t nodes_falsenodeids_t; Shape_t nodes_featureids_t; Shape_t nodes_missing_value_tracks_true_t; int* nodes_modes_t; Shape_t nodes_nodeids_t; Shape_t nodes_treeids_t; Shape_t nodes_truenodeids_t; int post_transform_t; Shape_t target_ids_t; Shape_t target_nodeids_t; Shape_t target_treeids_t;
        };
            
		Tensor* base_values;
		Tensor* nodes_hitrates;
		Tensor* nodes_values;
		Tensor* target_weights;
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string X; std::string Y;
        //parameter 
        Shape_t X_t; Shape_t Y_t; int aggregate_function_t; int n_targets_t; Shape_t nodes_falsenodeids_t; Shape_t nodes_featureids_t; Shape_t nodes_missing_value_tracks_true_t; int* nodes_modes_t; Shape_t nodes_nodeids_t; Shape_t nodes_treeids_t; Shape_t nodes_truenodeids_t; int post_transform_t; Shape_t target_ids_t; Shape_t target_nodeids_t; Shape_t target_treeids_t;

    public:
        TreeEnsembleRegressor(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            X = i[0];
            Y = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/treeensembleregressor.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({X_t, Y_t, aggregate_function_t, n_targets_t, nodes_falsenodeids_t, nodes_featureids_t, nodes_missing_value_tracks_true_t, nodes_modes_t, nodes_nodeids_t, nodes_treeids_t, nodes_truenodeids_t, post_transform_t, target_ids_t, target_nodeids_t, target_treeids_t }, tensor_dict[Y], tensor_dict[X]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["X"], X_t);
			convert_vec_param(a["Y"], Y_t);
			convert_vec_param(a["aggregate_function"], aggregate_function_t);
			convert_vec_param(a["n_targets"], n_targets_t);
			convert_vec_param(a["nodes_falsenodeids"], nodes_falsenodeids_t);
			convert_vec_param(a["nodes_featureids"], nodes_featureids_t);
			convert_vec_param(a["nodes_missing_value_tracks_true"], nodes_missing_value_tracks_true_t);
			convert_vec_param(a["nodes_modes"], nodes_modes_t);
			convert_vec_param(a["nodes_nodeids"], nodes_nodeids_t);
			convert_vec_param(a["nodes_treeids"], nodes_treeids_t);
			convert_vec_param(a["nodes_truenodeids"], nodes_truenodeids_t);
			convert_vec_param(a["post_transform"], post_transform_t);
			convert_vec_param(a["target_ids"], target_ids_t);
			convert_vec_param(a["target_nodeids"], target_nodeids_t);
			convert_vec_param(a["target_treeids"], target_treeids_t);   
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

    
        ~TreeEnsembleRegressor(){}

    };
}

#endif
