#ifndef SIGN_H
#define SIGN_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Calculate the sign of the given input tensor element-wise.
If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.

input: Input tensor
output: The sign of the input tensor computed element-wise. It has the same shape and type of the input.
*/

//Sign
//INPUTS:                   input_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Sign : public Layer {
        typedef struct {
            
			
            Shape_t input_i;
            
            Shape_t output_o;
            
        } binding_descriptor;

        
        std::string input_i;
        
        std::string output_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Sign(const std::string& name);
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _input_i, std::string _output_o); 

        ~Sign() {}
    };

}

#endif

