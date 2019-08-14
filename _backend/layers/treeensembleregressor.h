#ifndef TREEENSEMBLEREGRESSOR_H
#define TREEENSEMBLEREGRESSOR_H //TreeEnsembleRegressor
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      aggregate_function, base_values, n_targets, nodes_falsenodeids, nodes_featureids, nodes_hitrates, nodes_missing_value_tracks_true, nodes_modes, nodes_nodeids, nodes_treeids, nodes_truenodeids, nodes_values, post_transform, target_ids, target_nodeids, target_treeids, target_weights
//OPTIONAL_PARAMETERS_TYPE: int, Tensor*, int, Shape_t, Shape_t, Tensor*, Shape_t, Tensor*, Shape_t, Shape_t, Shape_t, Tensor*, int, Shape_t, Shape_t, Shape_t, Tensor*

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct TreeEnsembleRegressor_parameter_descriptor{    
        int aggregate_function; Tensor* base_values; int n_targets; Shape_t nodes_falsenodeids; Shape_t nodes_featureids; Tensor* nodes_hitrates; Shape_t nodes_missing_value_tracks_true; Tensor* nodes_modes; Shape_t nodes_nodeids; Shape_t nodes_treeids; Shape_t nodes_truenodeids; Tensor* nodes_values; int post_transform; Shape_t target_ids; Shape_t target_nodeids; Shape_t target_treeids; Tensor* target_weights;
    };   

    struct TreeEnsembleRegressor_input_desriptor{
        Tensor* X_input;
        
    };

    struct TreeEnsembleRegressor_output_descriptor{
        Tensor* Y_output;
        
    };

    struct TreeEnsembleRegressor_binding_descriptor{
        int aggregate_function; int n_targets; Shape_t nodes_falsenodeids; Shape_t nodes_featureids; Shape_t nodes_missing_value_tracks_true; Shape_t nodes_nodeids; Shape_t nodes_treeids; Shape_t nodes_truenodeids; int post_transform; Shape_t target_ids; Shape_t target_nodeids; Shape_t target_treeids;
		Shape_t base_values; Shape_t nodes_hitrates; Shape_t nodes_modes; Shape_t nodes_values; Shape_t target_weights;
        Shape_t X_input;
        
        Shape_t Y_output;
        
    };
}


namespace backend {

    class TreeEnsembleRegressor : public Layer {
        TreeEnsembleRegressor_parameter_descriptor parameters;
        TreeEnsembleRegressor_input_desriptor      input;
        TreeEnsembleRegressor_output_descriptor    output;
        TreeEnsembleRegressor_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, TreeEnsembleRegressor_binding_descriptor>* program;
        
    public:
        TreeEnsembleRegressor(std::string, TreeEnsembleRegressor_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~TreeEnsembleRegressor() {}

    };
}

//cpp stuff
namespace backend {    
   
    TreeEnsembleRegressor::TreeEnsembleRegressor(std::string n, TreeEnsembleRegressor_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, TreeEnsembleRegressor_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/treeensembleregressor.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* TreeEnsembleRegressor::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<TreeEnsembleRegressor, Layer>(m, "TreeEnsembleRegressor")
            .def("forward", &TreeEnsembleRegressor::forward);    
    }*/
}

#endif
