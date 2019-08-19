#ifndef GATHER_H
#define GATHER_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

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
        Gather();
    
        void forward() { program->run(); }
        
        void init( int _axis); 
        void bind(std::string _data_input, std::string _indices_input, std::string _output_output); 

        ~Gather() {}
    };

    
    void init_layer_Gather(py::module& m) {
        // py::class_(m, "Gather");
    }
    

}


#endif

