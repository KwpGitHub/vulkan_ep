#pragma once
#ifndef IF_H
#define IF_H 

#include "../layer.h"

/*
If conditional
input: Condition for the if
output: Values that are live-out to the enclosing scope. The return values in the `then_branch` and `else_branch` must be of the same shape and same data type.
*/

//If
//INPUTS:                   cond_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               else_branch, then_branch
//PARAMETER_TYPES:          int, int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class If : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        int m_else_branch; int m_then_branch;
        std::string m_cond_i;
        
        
        

        binding_descriptor   binding;
       

    public:
        If(std::string name);
        
        virtual void forward();        
        virtual void init( int _else_branch,  int _then_branch); 
        virtual void bind(std::string _cond_i); 
        virtual void build();

        ~If() {}
    };
   
}
#endif

