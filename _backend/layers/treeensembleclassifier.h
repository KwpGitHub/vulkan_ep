#ifndef TREEENSEMBLECLASSIFIER_H
#define TREEENSEMBLECLASSIFIER_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Tree Ensemble classifier.  Returns the top class for each of N inputs.<br>
    The attributes named 'nodes_X' form a sequence of tuples, associated by 
    index into the sequences, which must all be of equal length. These tuples
    define the nodes.<br>
    Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.
    A leaf may have multiple votes, where each vote is weighted by
    the associated class_weights index.<br>
    One and only one of classlabels_strings or classlabels_int64s
    will be defined. The class_ids are indices into this list.

input: Input of shape [N,F]
output: N, Top class for each point
output: The class score for each class, for each point, a tensor of shape [N,E].
//*/
//TreeEnsembleClassifier
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output, Z_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      base_values, class_ids, class_nodeids, class_treeids, class_weights, classlabels_int64s, classlabels_strings, nodes_falsenodeids, nodes_featureids, nodes_hitrates, nodes_missing_value_tracks_true, nodes_modes, nodes_nodeids, nodes_treeids, nodes_truenodeids, nodes_values, post_transform
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Shape_t, Shape_t, Shape_t, Tensor*, Shape_t, Tensor*, Shape_t, Shape_t, Tensor*, Shape_t, Tensor*, Shape_t, Shape_t, Shape_t, Tensor*, int

namespace py = pybind11;

//class stuff
namespace backend {   

    class TreeEnsembleClassifier : public Layer {
        typedef struct {
            Shape_t class_ids; Shape_t class_nodeids; Shape_t class_treeids; Shape_t classlabels_int64s; Shape_t nodes_falsenodeids; Shape_t nodes_featureids; Shape_t nodes_missing_value_tracks_true; Shape_t nodes_nodeids; Shape_t nodes_treeids; Shape_t nodes_truenodeids; int post_transform;
			Shape_t base_values; Shape_t class_weights; Shape_t classlabels_strings; Shape_t nodes_hitrates; Shape_t nodes_modes; Shape_t nodes_values;
            Shape_t X_input;
            
            Shape_t Y_output; Shape_t Z_output;
            
        } binding_descriptor;

        Shape_t class_ids; Shape_t class_nodeids; Shape_t class_treeids; Shape_t classlabels_int64s; Shape_t nodes_falsenodeids; Shape_t nodes_featureids; Shape_t nodes_missing_value_tracks_true; Shape_t nodes_nodeids; Shape_t nodes_treeids; Shape_t nodes_truenodeids; int post_transform; std::string base_values; std::string class_weights; std::string classlabels_strings; std::string nodes_hitrates; std::string nodes_modes; std::string nodes_values;
        std::string X_input;
        
        std::string Y_output; std::string Z_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        TreeEnsembleClassifier(std::string n, Shape_t class_ids, Shape_t class_nodeids, Shape_t class_treeids, Shape_t classlabels_int64s, Shape_t nodes_falsenodeids, Shape_t nodes_featureids, Shape_t nodes_missing_value_tracks_true, Shape_t nodes_nodeids, Shape_t nodes_treeids, Shape_t nodes_truenodeids, int post_transform);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string base_values, std::string class_weights, std::string classlabels_strings, std::string nodes_hitrates, std::string nodes_modes, std::string nodes_values, std::string X_input, std::string Y_output, std::string Z_output); 

        ~TreeEnsembleClassifier() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    TreeEnsembleClassifier::TreeEnsembleClassifier(std::string n, Shape_t class_ids, Shape_t class_nodeids, Shape_t class_treeids, Shape_t classlabels_int64s, Shape_t nodes_falsenodeids, Shape_t nodes_featureids, Shape_t nodes_missing_value_tracks_true, Shape_t nodes_nodeids, Shape_t nodes_treeids, Shape_t nodes_truenodeids, int post_transform) : Layer(n) { }
       
    vuh::Device* TreeEnsembleClassifier::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void TreeEnsembleClassifier::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
  		binding.Z_output = tensor_dict[Z_output]->shape();
 
		binding.class_ids = class_ids;
  		binding.class_nodeids = class_nodeids;
  		binding.class_treeids = class_treeids;
  		binding.classlabels_int64s = classlabels_int64s;
  		binding.nodes_falsenodeids = nodes_falsenodeids;
  		binding.nodes_featureids = nodes_featureids;
  		binding.nodes_missing_value_tracks_true = nodes_missing_value_tracks_true;
  		binding.nodes_nodeids = nodes_nodeids;
  		binding.nodes_treeids = nodes_treeids;
  		binding.nodes_truenodeids = nodes_truenodeids;
  		binding.post_transform = post_transform;
  		binding.base_values = tensor_dict[base_values]->shape();
  		binding.class_weights = tensor_dict[class_weights]->shape();
  		binding.classlabels_strings = tensor_dict[classlabels_strings]->shape();
  		binding.nodes_hitrates = tensor_dict[nodes_hitrates]->shape();
  		binding.nodes_modes = tensor_dict[nodes_modes]->shape();
  		binding.nodes_values = tensor_dict[nodes_values]->shape();
 
    }
    
    void TreeEnsembleClassifier::call(std::string base_values, std::string class_weights, std::string classlabels_strings, std::string nodes_hitrates, std::string nodes_modes, std::string nodes_values, std::string X_input, std::string Y_output, std::string Z_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/treeensembleclassifier.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[base_values]->data(), *tensor_dict[class_weights]->data(), *tensor_dict[classlabels_strings]->data(), *tensor_dict[nodes_hitrates]->data(), *tensor_dict[nodes_modes]->data(), *tensor_dict[nodes_values]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data(), *tensor_dict[Z_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<TreeEnsembleClassifier, Layer>(m, "TreeEnsembleClassifier")
            .def(py::init<std::string, Shape_t, Shape_t, Shape_t, Shape_t, Shape_t, Shape_t, Shape_t, Shape_t, Shape_t, Shape_t, int> ())
            .def("forward", &TreeEnsembleClassifier::forward)
            .def("init", &TreeEnsembleClassifier::init)
            .def("call", (void (TreeEnsembleClassifier::*) (std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string)) &TreeEnsembleClassifier::call);
    }
}

#endif

/* PYTHON STUFF

*/

