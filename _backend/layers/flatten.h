#ifndef FLATTEN_H
#define FLATTEN_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).

input: A tensor of rank >= axis.
output: A 2D tensor with the contents of the input tensor, with input dimensions up to axis flattened to the outer dimension of the output and remaining input dimensions flattened into the inner dimension of the output.
*/

//Flatten
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis
//OPTIONAL_PARAMETERS_TYPE: int

//class stuff
namespace backend {   

    class Flatten : public Layer {
        typedef struct {
            int axis;
			
            Shape_t input_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        int axis;
        std::string input_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Flatten();
    
        void forward() { program->run(); }
        
        void init( int _axis); 
        void bind(std::string _input_input, std::string _output_output); 

        ~Flatten() {}
    };

    
    void init_layer_Flatten(py::module& m) {
        // py::class_(m, "Flatten");
    }
    

}


#endif

