#pragma once
#ifndef SHAPE_H
#define SHAPE_H 

#include "../layer.h"

/*

Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.

input: An input tensor.
output: Shape of the input tensor
*/

//Shape
//INPUTS:                   data_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   shape_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Shape : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        
        std::string m_data_i;
        
        std::string m_shape_o;
        

        binding_descriptor   binding;
       

    public:
        Shape(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _data_i, std::string _shape_o); 
        virtual void build();

        ~Shape() {}
    };
   
}
#endif

