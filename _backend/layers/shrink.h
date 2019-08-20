#ifndef SHRINK_H
#define SHRINK_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Shrink takes one input data (Tensor<numeric>) and produces one Tensor output,
having same datatype and shape with input. It has two attributes, lambd and
bias. The formula of this operator is: If x < -lambd, y = x + bias;
If x > lambd, y = x - bias; Otherwise, y = 0.

input: The input data as Tensor.
output: The output.
*/

//Shrink
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      bias, lambd
//OPTIONAL_PARAMETERS_TYPE: float, float

//class stuff
namespace backend {   

    class Shrink : public Layer {
        typedef struct {
            float bias; float lambd;
			
            Shape_t input_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        float bias; float lambd;
        std::string input_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Shrink(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( float _bias,  float _lambd); 
        void bind(std::string _input_input, std::string _output_output); 

        ~Shrink() {}
    };

}

#endif

