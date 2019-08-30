#ifndef SLICE_H
#define SLICE_H 

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
//INPUTS:                   data_i, starts_i, ends_i
//OPTIONAL_INPUTS:          axes_i, steps_i
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Slice : public backend::Layer {
        typedef struct {
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        
        std::string m_data_i; std::string m_starts_i; std::string m_ends_i;
        std::string m_axes_i; std::string m_steps_i;
        std::string m_output_o;
        

        binding_descriptor   binding;
       

    public:
        Slice(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _data_i, std::string _starts_i, std::string _ends_i, std::string _axes_i, std::string _steps_i, std::string _output_o); 
        virtual void build();

        ~Slice() {}
    };
   
}
#endif

