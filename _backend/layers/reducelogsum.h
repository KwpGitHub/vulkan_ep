#ifndef REDUCELOGSUM_H
#define REDUCELOGSUM_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Computes the log sum of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.
input: An input tensor.
output: Reduced output tensor.
*/

//ReduceLogSum
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   reduced_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axes, keepdims
//OPTIONAL_PARAMETERS_TYPE: Shape_t, int

//class stuff
namespace backend {   

    class ReduceLogSum : public Layer {
        typedef struct {
            Shape_t axes; int keepdims;
			
            Shape_t data_input;
            
            Shape_t reduced_output;
            
        } binding_descriptor;

        Shape_t axes; int keepdims;
        std::string data_input;
        
        std::string reduced_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ReduceLogSum();
    
        void forward() { program->run(); }
        
        void init( Shape_t _axes,  int _keepdims); 
        void bind(std::string _data_input, std::string _reduced_output); 

        ~ReduceLogSum() {}
    };

    
    void init_layer_ReduceLogSum(py::module& m) {
        // py::class_(m, "ReduceLogSum");
    }
    

}


#endif

