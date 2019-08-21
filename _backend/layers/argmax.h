#ifndef ARGMAX_H
#define ARGMAX_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Computes the indices of the max elements of the input tensor's element along the 
provided axis. The resulted tensor has the same rank as the input if keepdims equal 1.
If keepdims equal 0, then the resulted tensor have the reduced dimension pruned. 
The type of the output tensor is integer.
input: An input tensor.
output: Reduced output tensor with integer data type.
*/

//ArgMax
//INPUTS:                   data_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   reduced_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis, keepdims
//OPTIONAL_PARAMETERS_TYPE: int, int


//class stuff
namespace backend {   

    class ArgMax : public Layer {
        typedef struct {
            int axis; int keepdims;
			
            Shape_t data_i;
            
            Shape_t reduced_o;
            
        } binding_descriptor;

        int axis; int keepdims;
        std::string data_i;
        
        std::string reduced_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ArgMax(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( int _axis,  int _keepdims); 
        virtual void bind(std::string _data_i, std::string _reduced_o); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/argmax.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[reduced_o]->data());
        }

        ~ArgMax() {}
    };
   
}
#endif

