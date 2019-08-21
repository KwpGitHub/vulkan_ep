#ifndef REDUCESUM_H
#define REDUCESUM_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Computes the sum of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.
input: An input tensor.
output: Reduced output tensor.
*/

//ReduceSum
//INPUTS:                   data_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   reduced_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axes, keepdims
//OPTIONAL_PARAMETERS_TYPE: Shape_t, int


//class stuff
namespace backend {   

    class ReduceSum : public Layer {
        typedef struct {
            Shape_t axes; int keepdims;
			
            Shape_t data_i;
            
            Shape_t reduced_o;
            
        } binding_descriptor;

        Shape_t axes; int keepdims;
        std::string data_i;
        
        std::string reduced_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ReduceSum(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( Shape_t _axes,  int _keepdims); 
        virtual void bind(std::string _data_i, std::string _reduced_o); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/reducesum.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[reduced_o]->data());
        }

        ~ReduceSum() {}
    };
   
}
#endif

