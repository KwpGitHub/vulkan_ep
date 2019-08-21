#ifndef UNSQUEEZE_H
#define UNSQUEEZE_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Insert single-dimensional entries to the shape of a tensor.
Takes one required argument `axes`, a list of dimensions that will be inserted.
Dimension indices in `axes` are as seen in the output tensor. For example:
  Given a tensor such that tensor with shape [3, 4, 5], then
  Unsqueeze(tensor, axes=[0, 4]) has shape [1, 3, 4, 5, 1]

input: Original tensor
output: Reshaped tensor with same data as input.
*/

//Unsqueeze
//INPUTS:                   data_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   expanded_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               axes
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace backend {   

    class Unsqueeze : public Layer {
        typedef struct {
            Shape_t axes;
			
            Shape_t data_i;
            
            Shape_t expanded_o;
            
        } binding_descriptor;

        Shape_t axes;
        std::string data_i;
        
        std::string expanded_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Unsqueeze(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( Shape_t _axes); 
        virtual void bind(std::string _data_i, std::string _expanded_o); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/unsqueeze.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[expanded_o]->data());
        }

        ~Unsqueeze() {}
    };
   
}
#endif

