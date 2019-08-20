#ifndef CONSTANT_H
#define CONSTANT_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*
A constant tensor.

output: Output tensor containing the same value of the provided tensor.
*/

//Constant
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               value
//PARAMETER_TYPES:          Tensor*
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Constant : public Layer {
        typedef struct {
            
			Shape_t value;
            
            
            Shape_t output_output;
            
        } binding_descriptor;

        std::string value;
        
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Constant(const std::string& name);
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _value, std::string _output_output); 

        ~Constant() {}
    };

}

#endif

