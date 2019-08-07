#ifndef SVMCLASSIFIER_H
#define SVMCLASSIFIER_H //SVMClassifier

//INPUTS:                   X
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y, Z
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      classlabels_ints, classlabels_strings, coefficients, kernel_params, kernel_type, post_transform, prob_a, prob_b, rho, support_vectors, vectors_per_class
//OPTIONAL_PARAMETERS_TYPE: INTS, STRINGS, FLOATS, FLOATS, STRING, STRING, FLOATS, FLOATS, FLOATS, FLOATS, INTS

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class SVMClassifier : public Layer {
        
        vuh::Device* _get_device();

        struct Params{ };
        vuh::Program<Specs, Params>* program;

    public:
        SVMClassifier(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
         
         //std::vector<uint32_t> output_shape();
   
        ~SVMClassifier(){}
    };
}


namespace backend {    
    SVMClassifier::SVMClassifier(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/svmclassifier.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* SVMClassifier::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }


};

#endif
