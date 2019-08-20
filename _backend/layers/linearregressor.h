#ifndef LINEARREGRESSOR_H
#define LINEARREGRESSOR_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

    Generalized linear regression evaluation.<br>
    If targets is set to 1 (default) then univariate regression is performed.<br>
    If targets is set to M then M sets of coefficients must be passed in as a sequence
    and M results will be output for each input n in N.<br>
    The coefficients array is of length n, and the coefficients for each target are contiguous.
    Intercepts are optional but if provided must match the number of targets.

input: Data to be regressed.
output: Regression outputs (one per target, per example).
*/

//LinearRegressor
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
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
            Shape_t X_i;
            
            Shape_t Y_o;
            
        } binding_descriptor;

        int post_transform; int targets; std::string coefficients; std::string intercepts;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        LinearRegressor(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( int _post_transform,  int _targets); 
        void bind(std::string _coefficients, std::string _intercepts, std::string _X_i, std::string _Y_o); 

        ~LinearRegressor() {}
    };

}

#endif

