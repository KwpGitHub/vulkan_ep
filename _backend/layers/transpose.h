#ifndef TRANSPOSE_H
#define TRANSPOSE_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Transpose the input tensor similar to numpy.transpose. For example, when
perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
will be (2, 1, 3).

input: An input tensor.
output: Transposed output.

*/
//Transpose
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   transposed_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      perm
//OPTIONAL_PARAMETERS_TYPE: Shape_t

namespace py = pybind11;

//class stuff
namespace backend {   

    class Transpose : public Layer {
        typedef struct {    
            Shape_t perm;
        } parameter_descriptor;  

        typedef struct {
            Tensor* data_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* transposed_output;
            
        } output_descriptor;

        typedef struct {
            Shape_t perm;
		
            Shape_t data_input;
            
            Shape_t transposed_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Transpose(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~Transpose() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Transpose::Transpose(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/transpose.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* Transpose::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Transpose::init() {
		binding.data_input = input.data_input->shape();
 
		binding.transposed_output = output.transposed_output->shape();
 
		binding.perm = parameters.perm;
 
        program->bind(binding, *input.data_input->data(), *output.transposed_output->data());
    }
    
    void Transpose::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Transpose, Layer>(m, "Transpose")
            .def("forward", &Transpose::forward);    
    }
}*/

#endif
