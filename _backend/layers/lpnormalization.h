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
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
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
			
            Shape_t input_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        int axis; int p;
        std::string input_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        LpNormalization();
    
        void forward() { program->run(); }
        
        void init( int _axis,  int _p); 
        void bind(std::string _input_input, std::string _output_output); 

        ~LpNormalization() {}
    };

    
    void init_layer_LpNormalization(py::module& m) {
        // py::class_(m, "LpNormalization");
    }
    

}


#endif

