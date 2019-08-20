#ifndef CONSTANTOFSHAPE_H
#define CONSTANTOFSHAPE_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Generate a tensor with given value and shape.

input: 1D tensor. The shape of the expected output tensor. If empty tensor is given, the output would be a scalar.
output: Output tensor of shape specified by 'input'.If attribute 'value' is specified, the value and datatype of the output tensor is taken from 'value'.If attribute 'value' is not specified, the value in the output defaults to 0, and the datatype defaults to float32.
*/

//ConstantOfShape
//INPUTS:                   input_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      value
//OPTIONAL_PARAMETERS_TYPE: Tensor*

//class stuff
namespace backend {   

    class ConstantOfShape : public Layer {
        typedef struct {
            
			Shape_t value;
            Shape_t input_i;
            
            Shape_t output_o;
            
        } binding_descriptor;

        std::string value;
        std::string input_i;
        
        std::string output_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ConstantOfShape(const std::string& name);
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _value, std::string _input_i, std::string _output_o); 

        ~ConstantOfShape() {}
    };

}

#endif

