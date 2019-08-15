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

*/
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
            Tensor* base_values; Shape_t class_ids; Shape_t class_nodeids; Shape_t class_treeids; Tensor* class_weights; Shape_t classlabels_int64s; Tensor* classlabels_strings; Shape_t nodes_falsenodeids; Shape_t nodes_featureids; Tensor* nodes_hitrates; Shape_t nodes_missing_value_tracks_true; Tensor* nodes_modes; Shape_t nodes_nodeids; Shape_t nodes_treeids; Shape_t nodes_truenodeids; Tensor* nodes_values; int post_transform;
        } parameter_descriptor;  

        typedef struct {
            Tensor* X_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* Y_output; Tensor* Z_output;
            
        } output_descriptor;

        typedef struct {
            Shape_t class_ids; Shape_t class_nodeids; Shape_t class_treeids; Shape_t classlabels_int64s; Shape_t nodes_falsenodeids; Shape_t nodes_featureids; Shape_t nodes_missing_value_tracks_true; Shape_t nodes_nodeids; Shape_t nodes_treeids; Shape_t nodes_truenodeids; int post_transform;
		Shape_t base_values; Shape_t class_weights; Shape_t classlabels_strings; Shape_t nodes_hitrates; Shape_t nodes_modes; Shape_t nodes_values;
            Shape_t X_input;
            
            Shape_t Y_output; Shape_t Z_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        TreeEnsembleClassifier(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~TreeEnsembleClassifier() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    TreeEnsembleClassifier::TreeEnsembleClassifier(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/treeensembleclassifier.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* TreeEnsembleClassifier::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void TreeEnsembleClassifier::init() {
		binding.X_input = input.X_input->shape();
 
		binding.Y_output = output.Y_output->shape();
  		binding.Z_output = output.Z_output->shape();
 
		binding.class_ids = parameters.class_ids;
  		binding.class_nodeids = parameters.class_nodeids;
  		binding.class_treeids = parameters.class_treeids;
  		binding.classlabels_int64s = parameters.classlabels_int64s;
  		binding.nodes_falsenodeids = parameters.nodes_falsenodeids;
  		binding.nodes_featureids = parameters.nodes_featureids;
  		binding.nodes_missing_value_tracks_true = parameters.nodes_missing_value_tracks_true;
  		binding.nodes_nodeids = parameters.nodes_nodeids;
  		binding.nodes_treeids = parameters.nodes_treeids;
  		binding.nodes_truenodeids = parameters.nodes_truenodeids;
  		binding.post_transform = parameters.post_transform;
  		binding.base_values = parameters.base_values->shape();
  		binding.class_weights = parameters.class_weights->shape();
  		binding.classlabels_strings = parameters.classlabels_strings->shape();
  		binding.nodes_hitrates = parameters.nodes_hitrates->shape();
  		binding.nodes_modes = parameters.nodes_modes->shape();
  		binding.nodes_values = parameters.nodes_values->shape();
 
        program->bind(binding, *parameters.base_values->data(), *parameters.class_weights->data(), *parameters.classlabels_strings->data(), *parameters.nodes_hitrates->data(), *parameters.nodes_modes->data(), *parameters.nodes_values->data(), *input.X_input->data(), *output.Y_output->data(), *output.Z_output->data());
    }
    
    void TreeEnsembleClassifier::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<TreeEnsembleClassifier, Layer>(m, "TreeEnsembleClassifier")
            .def("forward", &TreeEnsembleClassifier::forward);    
    }
}*/

#endif
