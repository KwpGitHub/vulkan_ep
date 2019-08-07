#ifndef TREEENSEMBLECLASSIFIER_H
#define TREEENSEMBLECLASSIFIER_H //TreeEnsembleClassifier

//INPUTS:                   X
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y, Z
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      base_values, class_ids, class_nodeids, class_treeids, class_weights, classlabels_int64s, classlabels_strings, nodes_falsenodeids, nodes_featureids, nodes_hitrates, nodes_missing_value_tracks_true, nodes_modes, nodes_nodeids, nodes_treeids, nodes_truenodeids, nodes_values, post_transform
//OPTIONAL_PARAMETERS_TYPE: FLOATS, INTS, INTS, INTS, FLOATS, INTS, STRINGS, INTS, INTS, FLOATS, INTS, STRINGS, INTS, INTS, INTS, FLOATS, STRING

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class TreeEnsembleClassifier : public Layer {
        
        vuh::Device* _get_device();

        struct Params{ };
        vuh::Program<Specs, Params>* program;

    public:
        TreeEnsembleClassifier(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
         
         //std::vector<uint32_t> output_shape();
   
        ~TreeEnsembleClassifier(){}
    };
}


namespace backend {    
    TreeEnsembleClassifier::TreeEnsembleClassifier(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/treeensembleclassifier.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* TreeEnsembleClassifier::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }


};

#endif
