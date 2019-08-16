#include "../layer.h"
#ifndef SVMCLASSIFIER_H
#define SVMCLASSIFIER_H 
/*

    Support Vector Machine classifier

input: Data to be classified.
output: Classification outputs (one class per example).
output: Class scores (one per class per example), if prob_a and prob_b are provided they are probabilities for each class, otherwise they are raw scores.
//*/
//SVMClassifier
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output, Z_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      classlabels_ints, classlabels_strings, coefficients, kernel_params, kernel_type, post_transform, prob_a, prob_b, rho, support_vectors, vectors_per_class
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*, Tensor*, Tensor*, int, int, Tensor*, Tensor*, Tensor*, Tensor*, Shape_t

//class stuff
namespace backend {   

    class SVMClassifier : public Layer {
        typedef struct {
            Shape_t classlabels_ints; int kernel_type; int post_transform; Shape_t vectors_per_class;
			Shape_t classlabels_strings; Shape_t coefficients; Shape_t kernel_params; Shape_t prob_a; Shape_t prob_b; Shape_t rho; Shape_t support_vectors;
            Shape_t X_input;
            
            Shape_t Y_output; Shape_t Z_output;
            
        } binding_descriptor;

        Shape_t classlabels_ints; int kernel_type; int post_transform; Shape_t vectors_per_class; std::string classlabels_strings; std::string coefficients; std::string kernel_params; std::string prob_a; std::string prob_b; std::string rho; std::string support_vectors;
        std::string X_input;
        
        std::string Y_output; std::string Z_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        SVMClassifier(std::string n, Shape_t classlabels_ints, int kernel_type, int post_transform, Shape_t vectors_per_class);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string classlabels_strings, std::string coefficients, std::string kernel_params, std::string prob_a, std::string prob_b, std::string rho, std::string support_vectors, std::string X_input, std::string Y_output, std::string Z_output); 

        ~SVMClassifier() {}

    };
    
}

#endif

