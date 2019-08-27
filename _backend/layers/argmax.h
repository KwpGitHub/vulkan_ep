#ifndef ARGMAX_H
#define ARGMAX_H 

#include "../layer.h"

/*

Computes the indices of the max elements of the input tensor's element along the 
provided axis. The resulted tensor has the same rank as the input if keepdims equal 1.
If keepdims equal 0, then the resulted tensor have the reduced dimension pruned. 
The type of the output tensor is integer.
input: An input tensor.
output: Reduced output tensor with integer data type.
*/

//ArgMax
//INPUTS:                   data_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   reduced_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis, keepdims
//OPTIONAL_PARAMETERS_TYPE: int, int


//class stuff
namespace layers {   

    class ArgMax : public backend::Layer {
        typedef struct {          
            backend::Shape_t data_i;
            
            backend::Shape_t reduced_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        int axis; int keepdims;
        std::string data_i;
        
        std::string reduced_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        ArgMax(std::string name);
        
        virtual void forward();        
        virtual void init( int _axis,  int _keepdims); 
        virtual void bind(std::string _data_i, std::string _reduced_o); 
        virtual void build();

        ~ArgMax() {}
    };
   
}
#endif

