#ifndef THRESHOLDEDRELU_H
#define THRESHOLDEDRELU_H 

#include "../layer.h"

/*

ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
is applied to the tensor elementwise.

input: Input tensor
output: Output tensor
*/

//ThresholdedRelu
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      alpha
//OPTIONAL_PARAMETERS_TYPE: float


//class stuff
namespace layers {   

    class ThresholdedRelu : public backend::Layer {
        typedef struct {          
            backend::Shape_t X_i;
            
            backend::Shape_t Y_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        float alpha;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        ThresholdedRelu(std::string name);
        
        virtual void forward();        
        virtual void init( float _alpha); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~ThresholdedRelu() {}
    };
   
}
#endif

