#ifndef SHRINK_H
#define SHRINK_H 

#include "../layer.h"

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
namespace layers {   

    class Shrink : public backend::Layer {
        typedef struct {          
            backend::Shape_t input_i;
            
            backend::Shape_t output_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        float bias; float lambd;
        std::string input_i;
        
        std::string output_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        Shrink(std::string name);
        
        virtual void forward();        
        virtual void init( float _bias,  float _lambd); 
        virtual void bind(std::string _input_i, std::string _output_o); 
        virtual void build();

        ~Shrink() {}
    };
   
}
#endif

