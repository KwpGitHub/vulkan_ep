#ifndef RESHAPE_H
#define RESHAPE_H 

#include "../layer.h"

/*

Reshape the input tensor similar to numpy.reshape.
First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor).
input: An input tensor.
input: Specified shape for output.
output: Reshaped data.
*/

//Reshape
//INPUTS:                   data_i, shape_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   reshaped_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Reshape : public backend::Layer {
        typedef struct {          
            backend::Shape_t data_i; backend::Shape_t shape_i;
            
            backend::Shape_t reshaped_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        
        std::string data_i; std::string shape_i;
        
        std::string reshaped_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        Reshape(std::string name);
        
        void forward() { program->run(); }
        
        virtual void init(); 
        virtual void bind(std::string _data_i, std::string _shape_i, std::string _reshaped_o); 
        virtual void build();

        ~Reshape() {}
    };
   
}
#endif

