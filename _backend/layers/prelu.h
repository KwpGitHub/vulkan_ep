#ifndef PRELU_H
#define PRELU_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check [the doc](Broadcasting.md).
input: Input tensor
input: Slope tensor. The shape of slope can be smaller then first input X; if so, its shape must be unidirectional broadcastable to X
output: Output tensor (same size as X)
*/

//PRelu
//INPUTS:                   X_input, slope_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class PRelu : public Layer {
        typedef struct {
            
			
            Shape_t X_input; Shape_t slope_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        
        std::string X_input; std::string slope_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        PRelu(const std::string& name);
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _X_input, std::string _slope_input, std::string _Y_output); 

        ~PRelu() {}
    };

}

#endif

