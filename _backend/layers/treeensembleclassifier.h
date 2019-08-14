#ifndef TREEENSEMBLECLASSIFIER_H
#define TREEENSEMBLECLASSIFIER_H //TreeEnsembleClassifier
#include <pybind11/pybind11.h>
#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output, Z_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      base_values, class_ids, class_nodeids, class_treeids, class_weights, classlabels_int64s, classlabels_strings, nodes_falsenodeids, nodes_featureids, nodes_hitrates, nodes_missing_value_tracks_true, nodes_modes, nodes_nodeids, nodes_treeids, nodes_truenodeids, nodes_values, post_transform
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Shape_t, Shape_t, Shape_t, Tensor*, Shape_t, Tensor*, Shape_t, Shape_t, Tensor*, Shape_t, Tensor*, Shape_t, Shape_t, Shape_t, Tensor*, int

namespace py = pybind11;

//descriptor stuff;
namespace backend {

    struct TreeEnsembleClassifier_parameter_descriptor{    
        Tensor* base_values; Shape_t class_ids; Shape_t class_nodeids; Shape_t class_treeids; Tensor* class_weights; Shape_t classlabels_int64s; Tensor* classlabels_strings; Shape_t nodes_falsenodeids; Shape_t nodes_featureids; Tensor* nodes_hitrates; Shape_t nodes_missing_value_tracks_true; Tensor* nodes_modes; Shape_t nodes_nodeids; Shape_t nodes_treeids; Shape_t nodes_truenodeids; Tensor* nodes_values; int post_transform;
    };   

    struct TreeEnsembleClassifier_input_desriptor{
        Tensor* X_input;
        
    };

    struct TreeEnsembleClassifier_output_descriptor{
        Tensor* Y_output; Tensor* Z_output;
        
    };

    struct TreeEnsembleClassifier_binding_descriptor{
        Shape_t class_ids; Shape_t class_nodeids; Shape_t class_treeids; Shape_t classlabels_int64s; Shape_t nodes_falsenodeids; Shape_t nodes_featureids; Shape_t nodes_missing_value_tracks_true; Shape_t nodes_nodeids; Shape_t nodes_treeids; Shape_t nodes_truenodeids; int post_transform;
		Shape_t base_values; Shape_t class_weights; Shape_t classlabels_strings; Shape_t nodes_hitrates; Shape_t nodes_modes; Shape_t nodes_values;
        Shape_t X_input;
        
        Shape_t Y_output; Shape_t Z_output;
        
    };
}


namespace backend {

    class TreeEnsembleClassifier : public Layer {
        TreeEnsembleClassifier_parameter_descriptor parameters;
        TreeEnsembleClassifier_input_desriptor      input;
        TreeEnsembleClassifier_output_descriptor    output;
        TreeEnsembleClassifier_binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, TreeEnsembleClassifier_binding_descriptor>* program;
        
    public:
        TreeEnsembleClassifier(std::string, TreeEnsembleClassifier_parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        void call() { program->bind(parameters); }
        ~TreeEnsembleClassifier() {}

    };
}

//cpp stuff
namespace backend {    
   
    TreeEnsembleClassifier::TreeEnsembleClassifier(std::string n, TreeEnsembleClassifier_parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, TreeEnsembleClassifier_binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/treeensembleclassifier.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }

  

    vuh::Device* TreeEnsembleClassifier::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
};


//python stuff
namespace backend{
    /*PYBIND11_MODULE(_backend, m) {
        py::class_<TreeEnsembleClassifier, Layer>(m, "TreeEnsembleClassifier")
            .def("forward", &TreeEnsembleClassifier::forward);    
    }*/
}

#endif
