#ifndef CONCAT_H
#define CONCAT_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*
Concatenate a list of tensors into a single tensor
input: List of tensors for concatenation
output: Concatenated tensor
*/

//Concat
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   concat_result_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               axis
//PARAMETER_TYPES:          int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Concat : public Layer {
        typedef struct {
            int axis;
			
            
            
            Shape_t concat_result_output;
            
        } binding_descriptor;

        int axis;
        
        
        std::string concat_result_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Concat(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( int _axis); 
        void bind(std::string _concat_result_output); 

        ~Concat() {}
    };

}

#endif

