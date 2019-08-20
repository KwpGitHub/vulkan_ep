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
//INPUTS:                   input_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
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
			
            Shape_t input_i;
            
            Shape_t output_o;
            
        } binding_descriptor;

        float bias; float lambd;
        std::string input_i;
        
        std::string output_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Shrink(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( float _bias,  float _lambd); 
        void bind(std::string _input_i, std::string _output_o); 

        ~Shrink() {}
    };

}

#endif

