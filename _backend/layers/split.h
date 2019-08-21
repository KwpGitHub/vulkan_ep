#ifndef SPLIT_H
#define SPLIT_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*
Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using argument 'split'.
Otherwise, the tensor is split to equal sized parts.

input: The tensor to split
output: One or more outputs forming list of tensors after splitting
*/

//Split
//INPUTS:                   input_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis, split
//OPTIONAL_PARAMETERS_TYPE: int, Shape_t


//class stuff
namespace backend {   

    class Split : public Layer {
        typedef struct {
            int axis; Shape_t split;
			
            Shape_t input_i;
            
            
            
        } binding_descriptor;

        int axis; Shape_t split;
        std::string input_i;
        
        
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Split(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( int _axis,  Shape_t _split); 
        virtual void bind(std::string _input_i); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/split.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[input_i]->data());
        }

        ~Split() {}
    };
   
}
#endif

