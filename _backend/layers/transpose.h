#ifndef TRANSPOSE_H
#define TRANSPOSE_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Transpose the input tensor similar to numpy.transpose. For example, when
perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
will be (2, 1, 3).

input: An input tensor.
output: Transposed output.
*/

//Transpose
//INPUTS:                   data_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   transposed_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      perm
//OPTIONAL_PARAMETERS_TYPE: Shape_t


//class stuff
namespace backend {   

    class Transpose : public Layer {
        typedef struct {
            Shape_t perm;
			
            Shape_t data_i;
            
            Shape_t transposed_o;
            
        } binding_descriptor;

        Shape_t perm;
        std::string data_i;
        
        std::string transposed_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Transpose(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( Shape_t _perm); 
        virtual void bind(std::string _data_i, std::string _transposed_o); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/transpose.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[transposed_o]->data());
        }

        ~Transpose() {}
    };
   
}
#endif

