#ifndef SINH_H
#define SINH_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Calculates the hyperbolic sine of the given input tensor element-wise.

input: Input tensor
output: The hyperbolic sine values of the input tensor computed element-wise
*/

//Sinh
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Sinh : public Layer {
        typedef struct {
            
			
            Shape_t input_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        
        std::string input_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Sinh();
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _input_input, std::string _output_output); 

        ~Sinh() {}
    };

    
    void init_layer_Sinh(py::module& m) {
        // py::class_(m, "Sinh");
    }
    

}


#endif

