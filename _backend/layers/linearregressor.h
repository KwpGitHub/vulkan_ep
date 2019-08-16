#include "../layer.h"
#ifndef LINEARREGRESSOR_H
#define LINEARREGRESSOR_H 
/*

    Generalized linear regression evaluation.<br>
    If targets is set to 1 (default) then univariate regression is performed.<br>
    If targets is set to M then M sets of coefficients must be passed in as a sequence
    and M results will be output for each input n in N.<br>
    The coefficients array is of length n, and the coefficients for each target are contiguous.
    Intercepts are optional but if provided must match the number of targets.

input: Data to be regressed.
output: Regression outputs (one per target, per example).
//*/
//LinearRegressor
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      coefficients, intercepts, post_transform, targets
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Tensor*, int, int

//class stuff
namespace backend {   

    class LinearRegressor : public Layer {
        typedef struct {
            int post_transform; int targets;
			Shape_t coefficients; Shape_t intercepts;
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        int post_transform; int targets; std::string coefficients; std::string intercepts;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        LinearRegressor(std::string n, int post_transform, int targets);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string coefficients, std::string intercepts, std::string X_input, std::string Y_output); 

        ~LinearRegressor() {}

    };
    
}

#endif

