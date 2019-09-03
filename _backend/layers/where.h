#pragma once
#ifndef WHERE_H
#define WHERE_H 

#include "../layer.h"

/*

    Return elements, either from X or Y, depending on condition
    (with Numpy-style broadcasting support).
    Where behaves like numpy.where with three parameters:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html

input: When True (nonzero), yield X, otherwise yield Y
input: values selected at indices where condition is True
input: values selected at indices where condition is False
output: Tensor of shape equal to the broadcasted shape of condition, X, and Y.
*/

//Where
//INPUTS:                   condition_i, X_i, Y_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Where : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        
        std::string m_condition_i; std::string m_X_i; std::string m_Y_i;
        
        std::string m_output_o;
        

        binding_descriptor   binding;
       

    public:
        Where(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _condition_i, std::string _X_i, std::string _Y_i, std::string _output_o); 
        virtual void build();

        ~Where() {}
    };
   
}
#endif

