#ifndef GATHER_H
#define GATHER_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
them in an output tensor of rank q + (r - 1).
Example 1:
  data = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  indices = [
      [0, 1],
      [1, 2],
  ]
  output = [
      [
          [1.0, 1.2],
          [2.3, 3.4],
      ],
      [
          [2.3, 3.4],
          [4.5, 5.7],
      ],
  ]
Example 2:
  data = [
      [1.0, 1.2, 1.9],
      [2.3, 3.4, 3.9],
      [4.5, 5.7, 5.9],
  ]
  indices = [
      [0, 2],
  ]
  axis = 1,
  output = [
      [
          [1.0, 1.9],
          [2.3, 3.9],
          [4.5, 5.9],
      ],
  ]

input: Tensor of rank r >= 1.
input: Tensor of int32/int64 indices, of any rank q.
output: Tensor of rank q + (r - 1).

*/
//Gather
//INPUTS:                   data_input, indices_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis
//OPTIONAL_PARAMETERS_TYPE: int

namespace py = pybind11;

//class stuff
namespace backend {   

    class Gather : public Layer {
        typedef struct {    
            int axis;
        } parameter_descriptor;  

        typedef struct {
            Tensor* data_input; Tensor* indices_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* output_output;
            
        } output_descriptor;

        typedef struct {
            int axis;
		
            Shape_t data_input; Shape_t indices_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Gather(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~Gather() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Gather::Gather(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/gather.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* Gather::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Gather::init() {
		binding.data_input = input.data_input->shape();
  		binding.indices_input = input.indices_input->shape();
 
		binding.output_output = output.output_output->shape();
 
		binding.axis = parameters.axis;
 
        program->bind(binding, *input.data_input->data(), *input.indices_input->data(), *output.output_output->data());
    }
    
    void Gather::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Gather, Layer>(m, "Gather")
            .def("forward", &Gather::forward);    
    }
}*/

#endif
