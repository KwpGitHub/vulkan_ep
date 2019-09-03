#pragma once
#ifndef MAX_H
#define MAX_H 

#include "../layer.h"

/*

Element-wise max of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

input: List of tensors for max.
output: Output tensor.
*/

//Max
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   max_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Max : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        
        
        
        std::string m_max_o;
        

        binding_descriptor   binding;
       

    public:
        Max(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _max_o); 
        virtual void build();

        ~Max() {}
    };
   
}
#endif

