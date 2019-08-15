#ifndef PAD_H
#define PAD_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Given `data` tensor, pads, mode, and value.
Example:
  Insert 0 pads to the beginning of the second dimension.
  data = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  pads = [0, 2, 0, 0]
  output = [
      [
          [0.0, 0.0, 1.0, 1.2],
          [0.0, 0.0, 2.3, 3.4],
          [0.0, 0.0, 4.5, 5.7],
      ],
  ]

input: Input tensor.
output: Tensor after padding.

*/
//Pad
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               pads
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      mode, value
//OPTIONAL_PARAMETERS_TYPE: int, float

namespace py = pybind11;

//class stuff
namespace backend {   

    class Pad : public Layer {
        typedef struct {    
            Shape_t pads; int mode; float value;
        } parameter_descriptor;  

        typedef struct {
            Tensor* data_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* output_output;
            
        } output_descriptor;

        typedef struct {
            Shape_t pads; int mode; float value;
		
            Shape_t data_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Pad(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~Pad() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Pad::Pad(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/pad.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* Pad::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Pad::init() {
		binding.data_input = input.data_input->shape();
 
		binding.output_output = output.output_output->shape();
 
		binding.pads = parameters.pads;
  		binding.mode = parameters.mode;
  		binding.value = parameters.value;
 
        program->bind(binding, *input.data_input->data(), *output.output_output->data());
    }
    
    void Pad::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Pad, Layer>(m, "Pad")
            .def("forward", &Pad::forward);    
    }
}*/

#endif
