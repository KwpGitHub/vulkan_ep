#ifndef SLICE_H
#define SLICE_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
Slices uses `starts`, `ends`, `axes` and `steps` inputs to specify the start and end
dimension and step for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represent number of elements before the end of that
dimension. If the value passed to start or end is larger than the `n` (the
number of elements in this dimension), it represents `n`. For slicing to the
end of a dimension with unknown size, it is recommended to pass in `INT_MAX`.
If a negative value is passed for step, it represents slicing backward.
If `axes` are omitted, they are set to `[0, ..., ndim-1]`.
If `steps` are omitted, they are set to `[1, ..., 1]` of length `len(starts)`
Example 1:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  axes = [0, 1]
  starts = [1, 0]
  ends = [2, 3]
  steps = [1, 2]
  result = [
      [5, 7],
  ]
Example 2:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 1000]
  result = [
      [2, 3, 4],
  ]

input: Tensor of data to extract slices from.
input: 1-D tensor of starting indices of corresponding axis in `axes`
input: 1-D tensor of ending indices (exclusive) of corresponding axis in `axes`
input: 1-D tensor of axes that `starts` and `ends` apply to.
input: 1-D tensor of slice step of corresponding axis in `axes`. Default to 1. 
output: Sliced data tensor.

*/
//Slice
//INPUTS:                   data_input, starts_input, ends_input
//OPTIONAL_INPUTS:          axes_input_opt, steps_input_opt
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class Slice : public Layer {
        typedef struct {    
            
        } parameter_descriptor;  

        typedef struct {
            Tensor* data_input; Tensor* starts_input; Tensor* ends_input;
            Tensor* axes_input_opt; Tensor* steps_input_opt;
        } input_desriptor;

        typedef struct {
            Tensor* output_output;
            
        } output_descriptor;

        typedef struct {
            
		
            Shape_t data_input; Shape_t starts_input; Shape_t ends_input;
            Shape_t axes_input_opt; Shape_t steps_input_opt;
            Shape_t output_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Slice(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~Slice() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Slice::Slice(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/slice.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* Slice::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Slice::init() {
		binding.data_input = input.data_input->shape();
  		binding.starts_input = input.starts_input->shape();
  		binding.ends_input = input.ends_input->shape();
  		binding.axes_input_opt = input.axes_input_opt->shape();
  		binding.steps_input_opt = input.steps_input_opt->shape();
 
		binding.output_output = output.output_output->shape();
 

        program->bind(binding, *input.data_input->data(), *input.starts_input->data(), *input.ends_input->data(), *input.axes_input_opt->data(), *input.steps_input_opt->data(), *output.output_output->data());
    }
    
    void Slice::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Slice, Layer>(m, "Slice")
            .def("forward", &Slice::forward);    
    }
}*/

#endif
