#ifndef COSH_H
#define COSH_H 

#include "../layer.h"

/*

Calculates the hyperbolic cosine of the given input tensor element-wise.

input: Input tensor
output: The hyperbolic cosine values of the input tensor computed element-wise
*/

//Cosh
//INPUTS:                   input_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Cosh : public backend::Layer {
        typedef struct {          
            backend::Shape_t input_i;
            
            backend::Shape_t output_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        
        std::string input_i;
        
        std::string output_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        Cosh(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _input_i, std::string _output_o); 
        virtual void build();

        ~Cosh() {}
    };
   
}
#endif

