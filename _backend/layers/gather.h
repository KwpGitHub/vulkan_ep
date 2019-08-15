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
//*/
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
			
            Shape_t data_input; Shape_t indices_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        int axis;
        std::string data_input; std::string indices_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Gather(std::string n, int axis);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string data_input, std::string indices_input, std::string output_output); 

        ~Gather() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Gather::Gather(std::string n, int axis) : Layer(n) { }
       
    vuh::Device* Gather::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Gather::init() {      
    
		binding.data_input = tensor_dict[data_input]->shape();
  		binding.indices_input = tensor_dict[indices_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.axis = axis;
 
    }
    
    void Gather::call(std::string data_input, std::string indices_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/gather.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[indices_input]->data(), *tensor_dict[output_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Gather, Layer>(m, "Gather")
            .def(py::init<std::string, int> ())
            .def("forward", &Gather::forward)
            .def("init", &Gather::init)
            .def("call", (void (Gather::*) (std::string, std::string, std::string)) &Gather::call);
    }
}

#endif

/* PYTHON STUFF

*/

