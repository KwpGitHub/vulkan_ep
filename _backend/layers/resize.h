#ifndef RESIZE_H
#define RESIZE_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Resize the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).

input: N-D tensor
input: The scale array along each dimension. It takes value greater than 0. If it's less than 1, it's sampling down, otherwise, it's upsampling. The number of elements of 'scales' should be the same as the rank of input 'X'.
output: N-D tensor after resizing

*/
//Resize
//INPUTS:                   X_input, scales_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      mode
//OPTIONAL_PARAMETERS_TYPE: int

namespace py = pybind11;

//class stuff
namespace backend {   

    class Resize : public Layer {
        typedef struct {    
            int mode;
        } parameter_descriptor;  

        typedef struct {
            Tensor* X_input; Tensor* scales_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* Y_output;
            
        } output_descriptor;

        typedef struct {
            int mode;
		
            Shape_t X_input; Shape_t scales_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Resize(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~Resize() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Resize::Resize(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/resize.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* Resize::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Resize::init() {
		binding.X_input = input.X_input->shape();
  		binding.scales_input = input.scales_input->shape();
 
		binding.Y_output = output.Y_output->shape();
 
		binding.mode = parameters.mode;
 
        program->bind(binding, *input.X_input->data(), *input.scales_input->data(), *output.Y_output->data());
    }
    
    void Resize::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Resize, Layer>(m, "Resize")
            .def("forward", &Resize::forward);    
    }
}*/

#endif
