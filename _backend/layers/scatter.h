#include "../layer.h"
#ifndef SCATTER_H
#define SCATTER_H 
/*

Given `data`, `updates` and `indices` input tensors of rank r >= 1, write the values provided by `updates` 
into the first input, `data`, along `axis` dimension of `data` (by default outer-most one as axis=0) at corresponding `indices`. 
For each entry in `updates`, the target index in `data` is specified by corresponding entry in `indices`
for dimension = axis, and index in source for dimension != axis. For instance, in a 2-D tensor case,
data[indices[i][j]][j] = updates[i][j] if axis = 0, or data[i][indices[i][j]] = updates[i][j] if axis = 1,
where i and j are loop counters from 0 up to the respective size in `updates` - 1.

Example 1:
  data = [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
  ]
  indices = [
      [1, 0, 2],
      [0, 2, 1],
  ]
  updates = [
      [1.0, 1.1, 1.2],
      [2.0, 2.1, 2.2],
  ]
  output = [
      [2.0, 1.1, 0.0]
      [1.0, 0.0, 2.2]
      [0.0, 2.1, 1.2]
  ]

Example 2:
  data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
  indices = [[1, 3]]
  updates = [[1.1, 2.1]]
  axis = 1
  output = [[1.0, 1.1, 3.0, 2.1, 5.0]]

input: Tensor of rank r >= 1.
input: Tensor of int32/int64 indices, of r >= 1 (same rank as input).
input: Tensor of rank r >=1 (same rank and shape as indices)
output: Tensor of rank r >= 1 (same rank as input).
//*/
//Scatter
//INPUTS:                   data_input, indices_input, updates_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis
//OPTIONAL_PARAMETERS_TYPE: int

//class stuff
namespace backend {   

    class Scatter : public Layer {
        typedef struct {
            int axis;
			
            Shape_t data_input; Shape_t indices_input; Shape_t updates_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        int axis;
        std::string data_input; std::string indices_input; std::string updates_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Scatter(std::string n);
    
        void forward() { program->run(); }
        
        void init( int _axis); 
        void bind(std::string _data_input, std::string _indices_input, std::string _updates_input, std::string _output_output); 

        ~Scatter() {}

    };
    
}

#endif

