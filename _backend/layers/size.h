#ifndef SIZE_H
#define SIZE_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.

input: An input tensor.
output: Total number of elements of the input tensor

*/
//Size
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   size_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class Size : public Layer {
        typedef struct {    
            
        } parameter_descriptor;  

        typedef struct {
            Tensor* data_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* size_output;
            
        } output_descriptor;

        typedef struct {
            
		
            Shape_t data_input;
            
            Shape_t size_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Size(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~Size() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Size::Size(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/size.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* Size::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Size::init() {
		binding.data_input = input.data_input->shape();
 
		binding.size_output = output.size_output->shape();
 

        program->bind(binding, *input.data_input->data(), *output.size_output->data());
    }
    
    void Size::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Size, Layer>(m, "Size")
            .def("forward", &Size::forward);    
    }
}*/

#endif
