#ifndef ACOSH_H
#define ACOSH_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Calculates the hyperbolic arccosine of the given input tensor element-wise.

input: Input tensor
output: The hyperbolic arccosine values of the input tensor computed element-wise
*/

//Acosh
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

    class Acosh : public Layer {
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
        Acosh(const std::string& name);
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _input_i, std::string _output_o); 

        ~Acosh() {}
    };

}

#endif

