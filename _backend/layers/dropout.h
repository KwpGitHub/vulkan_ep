#ifndef DROPOUT_H
#define DROPOUT_H 

#include "../layer.h"

/*

Dropout takes one input floating tensor and produces two tensor outputs,
output (floating tensor) and mask (`Tensor<bool>`). Depending on whether it is
in test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

input: The input data as Tensor.
output: The output.
output: The output mask.
*/

//Dropout
//INPUTS:                   data_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         mask_o
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      ratio
//OPTIONAL_PARAMETERS_TYPE: float


//class stuff
namespace layers {   

    class Dropout : public backend::Layer {
        typedef struct {          
            backend::Shape_t data_i;
            
            backend::Shape_t output_o;
            backend::Shape_t mask_o;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        float ratio;
        std::string data_i;
        
        std::string output_o;
        std::string mask_o;

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        Dropout(std::string name);
        
        virtual void forward();        
        virtual void init( float _ratio); 
        virtual void bind(std::string _data_i, std::string _output_o, std::string _mask_o); 
        virtual void build();

        ~Dropout() {}
    };
   
}
#endif

