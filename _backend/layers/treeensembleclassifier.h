#pragma once
#ifndef TREEENSEMBLECLASSIFIER_H
#define TREEENSEMBLECLASSIFIER_H 

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
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o, Z_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      base_values, class_ids, class_nodeids, class_treeids, class_weights, classlabels_int64s, classlabels_strings, nodes_falsenodeids, nodes_featureids, nodes_hitrates, nodes_missing_value_tracks_true, nodes_modes, nodes_nodeids, nodes_treeids, nodes_truenodeids, nodes_values, post_transform
//OPTIONAL_PARAMETERS_TYPE: std::vector<float>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<float>, std::vector<int>, std::vector<std::string>, std::vector<int>, std::vector<int>, std::vector<float>, std::vector<int>, std::vector<std::string>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<float>, std::string


//class stuff
namespace layers {   

    class TreeEnsembleClassifier : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<float> m_base_values; std::vector<int> m_class_ids; std::vector<int> m_class_nodeids; std::vector<int> m_class_treeids; std::vector<float> m_class_weights; std::vector<int> m_classlabels_int64s; std::vector<std::string> m_classlabels_strings; std::vector<int> m_nodes_falsenodeids; std::vector<int> m_nodes_featureids; std::vector<float> m_nodes_hitrates; std::vector<int> m_nodes_missing_value_tracks_true; std::vector<std::string> m_nodes_modes; std::vector<int> m_nodes_nodeids; std::vector<int> m_nodes_treeids; std::vector<int> m_nodes_truenodeids; std::vector<float> m_nodes_values; std::string m_post_transform;
        std::string m_X_i;
        
        std::string m_Y_o; std::string m_Z_o;
        

        binding_descriptor   binding;
       

    public:
        TreeEnsembleClassifier(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<float> _base_values,  std::vector<int> _class_ids,  std::vector<int> _class_nodeids,  std::vector<int> _class_treeids,  std::vector<float> _class_weights,  std::vector<int> _classlabels_int64s,  std::vector<std::string> _classlabels_strings,  std::vector<int> _nodes_falsenodeids,  std::vector<int> _nodes_featureids,  std::vector<float> _nodes_hitrates,  std::vector<int> _nodes_missing_value_tracks_true,  std::vector<std::string> _nodes_modes,  std::vector<int> _nodes_nodeids,  std::vector<int> _nodes_treeids,  std::vector<int> _nodes_truenodeids,  std::vector<float> _nodes_values,  std::string _post_transform); 
        virtual void bind(std::string _X_i, std::string _Y_o, std::string _Z_o); 
        virtual void build();

        ~TreeEnsembleClassifier() {}
    };
   
}
#endif

