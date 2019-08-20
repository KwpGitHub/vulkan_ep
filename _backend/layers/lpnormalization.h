#ifndef LPNORMALIZATION_H
#define LPNORMALIZATION_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Given a matrix, apply Lp-normalization along the provided axis.

input: Input matrix
output: Matrix after normalization
*/

//LpNormalization
//INPUTS:                   input_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis, p
//OPTIONAL_PARAMETERS_TYPE: int, int

//class stuff
namespace backend {   

    class LpNormalization : public Layer {
        typedef struct {
            int axis; int p;
			
            Shape_t input_i;
            
            Shape_t output_o;
            
        } binding_descriptor;

        int axis; int p;
        std::string input_i;
        
        std::string output_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        LpNormalization(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( int _axis,  int _p); 
        void bind(std::string _input_i, std::string _output_o); 

        ~LpNormalization() {}
    };

}

#endif

