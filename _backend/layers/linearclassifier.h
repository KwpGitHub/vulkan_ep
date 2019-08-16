#include "../layer.h"
#ifndef LINEARCLASSIFIER_H
#define LINEARCLASSIFIER_H 
/*

    Linear classifier

input: Data to be classified.
output: Classification outputs (one class per example).
output: Classification scores ([N,E] - one score for each class and example
//*/
//LinearClassifier
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output, Z_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               coefficients
//PARAMETER_TYPES:          Tensor*
//OPTIONAL_PARAMETERS:      classlabels_ints, classlabels_strings, intercepts, multi_class, post_transform
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*, Tensor*, int, int

//class stuff
namespace backend {   

    class LinearClassifier : public Layer {
        typedef struct {
            Shape_t classlabels_ints; int multi_class; int post_transform;
			Shape_t coefficients; Shape_t classlabels_strings; Shape_t intercepts;
            Shape_t X_input;
            
            Shape_t Y_output; Shape_t Z_output;
            
        } binding_descriptor;

        Shape_t classlabels_ints; int multi_class; int post_transform; std::string coefficients; std::string classlabels_strings; std::string intercepts;
        std::string X_input;
        
        std::string Y_output; std::string Z_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        LinearClassifier(std::string n, Shape_t classlabels_ints, int multi_class, int post_transform);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string coefficients, std::string classlabels_strings, std::string intercepts, std::string X_input, std::string Y_output, std::string Z_output); 

        ~LinearClassifier() {}

    };
    
}

#endif

