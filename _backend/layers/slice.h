#ifndef SLICE_H
#define SLICE_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

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

//class stuff
namespace backend {   

    class Slice : public Layer {
        typedef struct {
            
			
            Shape_t data_input; Shape_t starts_input; Shape_t ends_input;
            Shape_t axes_input_opt; Shape_t steps_input_opt;
            Shape_t output_output;
            
        } binding_descriptor;

        
        std::string data_input; std::string starts_input; std::string ends_input;
        std::string axes_input_opt; std::string steps_input_opt;
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Slice();
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _data_input, std::string _starts_input, std::string _ends_input, std::string _axes_input_opt, std::string _steps_input_opt, std::string _output_output); 

        ~Slice() {}
    };

    
    void init_layer_Slice(py::module& m) {
        // py::class_(m, "Slice");
    }
    

}


#endif

