#pragma once
#ifndef CONSTANTOFSHAPE_H
#define CONSTANTOFSHAPE_H 

#include "../layer.h"

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
//OPTIONAL_PARAMETERS_TYPE: std::vector<float>


//class stuff
namespace layers {   

    class ConstantOfShape : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<float> m_value;
        std::string m_input_i;
        
        std::string m_output_o;
        

        binding_descriptor   binding;
       

    public:
        ConstantOfShape(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<float> _value); 
        virtual void bind(std::string _input_i, std::string _output_o); 
        virtual void build();

        ~ConstantOfShape() {}
    };
   
}
#endif

