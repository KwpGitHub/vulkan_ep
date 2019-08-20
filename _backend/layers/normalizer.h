#ifndef NORMALIZER_H
#define NORMALIZER_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

    Normalize the input.  There are three normalization modes, which have the corresponding formulas,
    defined using element-wise infix operators '/' and '^' and tensor-wide functions 'max' and 'sum':<br>
<br>
    Max: Y = X / max(X)<br>
    L1:  Y = X / sum(X)<br>
    L2:  Y = sqrt(X^2 / sum(X^2)}<br>
    In all modes, if the divisor is zero, Y == X.
<br>
    For batches, that is, [N,C] tensors, normalization is done along the C axis. In other words, each row
    of the batch is normalized independently.

input: Data to be encoded, a tensor of shape [N,C] or [C]
output: Encoded output data
*/

//Normalizer
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      norm
//OPTIONAL_PARAMETERS_TYPE: int

//class stuff
namespace backend {   

    class Normalizer : public Layer {
        typedef struct {
            int norm;
			
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        int norm;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Normalizer(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( int _norm); 
        void bind(std::string _X_input, std::string _Y_output); 

        ~Normalizer() {}
    };

}

#endif

