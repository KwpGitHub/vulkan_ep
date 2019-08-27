#ifndef HARDSIGMOID_H
#define HARDSIGMOID_H 

#include "../layer.h"

/*

HardSigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
is applied to the tensor elementwise.

input: Input tensor
output: Output tensor
*/

//HardSigmoid
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      alpha, beta
//OPTIONAL_PARAMETERS_TYPE: float, float


//class stuff
namespace layers {   

    class HardSigmoid : public backend::Layer {
        typedef struct {          
            backend::Shape_t X_i;
            
            backend::Shape_t Y_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        float alpha; float beta;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        HardSigmoid(std::string name);
        
        virtual void forward();        
        virtual void init( float _alpha,  float _beta); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~HardSigmoid() {}
    };
   
}
#endif

