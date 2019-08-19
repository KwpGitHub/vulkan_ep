#ifndef DROPOUT_H
#define DROPOUT_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

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
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         mask_output_opt
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      ratio
//OPTIONAL_PARAMETERS_TYPE: float

//class stuff
namespace backend {   

    class Dropout : public Layer {
        typedef struct {
            float ratio;
			
            Shape_t data_input;
            
            Shape_t output_output;
            Shape_t mask_output_opt;
        } binding_descriptor;

        float ratio;
        std::string data_input;
        
        std::string output_output;
        std::string mask_output_opt;

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Dropout();
    
        void forward() { program->run(); }
        
        void init( float _ratio); 
        void bind(std::string _data_input, std::string _output_output, std::string _mask_output_opt); 

        ~Dropout() {}
    };

    
    void init_layer_Dropout(py::module& m) {
        // py::class_(m, "Dropout");
    }
    

}


#endif

