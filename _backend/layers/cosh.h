#ifndef COSH_H
#define COSH_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Calculates the hyperbolic cosine of the given input tensor element-wise.

input: Input tensor
output: The hyperbolic cosine values of the input tensor computed element-wise
*/

//Cosh
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

    class Cosh : public Layer {
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
        Cosh(const std::string& name);
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _input_i, std::string _output_o); 

        ~Cosh() {}
    };

}

#endif

