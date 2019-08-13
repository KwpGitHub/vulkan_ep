#ifndef TREEENSEMBLEREGRESSOR_H
#define TREEENSEMBLEREGRESSOR_H //TreeEnsembleRegressor

#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      aggregate_function, base_values, n_targets, nodes_falsenodeids, nodes_featureids, nodes_hitrates, nodes_missing_value_tracks_true, nodes_modes, nodes_nodeids, nodes_treeids, nodes_truenodeids, nodes_values, post_transform, target_ids, target_nodeids, target_treeids, target_weights
//OPTIONAL_PARAMETERS_TYPE: int, Tensor*, int, Shape_t, Shape_t, Tensor*, Shape_t, Tensor*, Shape_t, Shape_t, Shape_t, Tensor*, int, Shape_t, Shape_t, Shape_t, Tensor*



namespace backend {
    class TreeEnsembleRegressor : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int aggregate_function; int n_targets; Shape_t nodes_falsenodeids; Shape_t nodes_featureids; Shape_t nodes_missing_value_tracks_true; Shape_t nodes_nodeids; Shape_t nodes_treeids; Shape_t nodes_truenodeids; int post_transform; Shape_t target_ids; Shape_t target_nodeids; Shape_t target_treeids;
			Shape_t base_values; Shape_t nodes_hitrates; Shape_t nodes_modes; Shape_t nodes_values; Shape_t target_weights;
            //input
            Shape_t X_input;
            
            //output
            Shape_t Y_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        TreeEnsembleRegressor(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward() { program->run(); }
        
        int aggregate_function; Tensor* base_values; int n_targets; Shape_t nodes_falsenodeids; Shape_t nodes_featureids; Tensor* nodes_hitrates; Shape_t nodes_missing_value_tracks_true; Tensor* nodes_modes; Shape_t nodes_nodeids; Shape_t nodes_treeids; Shape_t nodes_truenodeids; Tensor* nodes_values; int post_transform; Shape_t target_ids; Shape_t target_nodeids; Shape_t target_treeids; Tensor* target_weights;
		Shape_t base_values_s; Shape_t nodes_hitrates_s; Shape_t nodes_modes_s; Shape_t nodes_values_s; Shape_t target_weights_s;
        //input
        std::string X_input;
        
        //output
        std::string Y_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~TreeEnsembleRegressor() {}
    };
}


namespace backend {    
    TreeEnsembleRegressor::TreeEnsembleRegressor(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
        program = new vuh::Program<Specs, Params>(*_get_device(), std::string(file_path + "/shaders/bin/treeensembleregressor.spv").c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind({aggregate_function, n_targets, nodes_falsenodeids, nodes_featureids, nodes_missing_value_tracks_true, nodes_nodeids, nodes_treeids, nodes_truenodeids, post_transform, target_ids, target_nodeids, target_treeids, base_values_s, nodes_hitrates_s, nodes_modes_s, nodes_values_s, target_weights_s, tensor_dict[X_input]->shape(), tensor_dict[Y_output]->shape()} 
                        , *base_values, *nodes_hitrates, *nodes_modes, *nodes_values, *target_weights
                        , tensor_dict[X_input], tensor_dict[Y_output] );
    }

    vuh::Device* TreeEnsembleRegressor::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
};

#endif
