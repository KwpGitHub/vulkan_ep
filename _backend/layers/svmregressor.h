#include "../layer.h"
#ifndef SVMREGRESSOR_H
#define SVMREGRESSOR_H 
/*

    Support Vector Machine regression prediction and one-class SVM anomaly detection.

input: Data to be regressed.
output: Regression outputs (one score per target per example).
//*/
//SVMRegressor
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      coefficients, kernel_params, kernel_type, n_supports, one_class, post_transform, rho, support_vectors
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Tensor*, int, int, int, int, Tensor*, Tensor*

//class stuff
namespace backend {   

    class SVMRegressor : public Layer {
        typedef struct {
            int kernel_type; int n_supports; int one_class; int post_transform;
			Shape_t coefficients; Shape_t kernel_params; Shape_t rho; Shape_t support_vectors;
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        int kernel_type; int n_supports; int one_class; int post_transform; std::string coefficients; std::string kernel_params; std::string rho; std::string support_vectors;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        SVMRegressor(std::string n);
    
        void forward() { program->run(); }
        
        void init( int _kernel_type,  int _n_supports,  int _one_class,  int _post_transform); 
        void bind(std::string _coefficients, std::string _kernel_params, std::string _rho, std::string _support_vectors, std::string _X_input, std::string _Y_output); 

        ~SVMRegressor() {}

    };
    
}

#endif

