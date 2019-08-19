#ifndef REVERSESEQUENCE_H
#define REVERSESEQUENCE_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Reverse batch of sequences having different lengths specified by `sequence_lens`.

For each slice i iterating on batch axis, the operator reverses the first sequence_lens[i] elements on time axis,
and copies elements whose index's beyond sequence_lens[i] to the output. So the output slice i contains reversed
sequences on the first sequence_lens[i] elements, then have original values copied for the other elements.

Example 1:
  input = [[0.0, 4.0, 8.0,  12.0],
           [1.0, 5.0, 9.0,  13.0],
           [2.0, 6.0, 10.0, 14.0],
           [3.0, 7.0, 11.0, 15.0]]
  sequence_lens = [4, 3, 2, 1]
  time_axis = 0
  batch_axis = 1

  output = [[3.0, 6.0, 9.0,  12.0],
            [2.0, 5.0, 8.0,  13.0],
            [1.0, 4.0, 10.0, 14.0],
            [0.0, 7.0, 11.0, 15.0]]

Example 2:
  input = [[0.0,  1.0,  2.0,  3.0 ],
           [4.0,  5.0,  6.0,  7.0 ],
           [8.0,  9.0,  10.0, 11.0],
           [12.0, 13.0, 14.0, 15.0]]
  sequence_lens = [1, 2, 3, 4]
  time_axis = 1
  batch_axis = 0

  output = [[0.0,  1.0,  2.0,  3.0 ],
            [5.0,  4.0,  6.0,  7.0 ],
            [10.0, 9.0,  8.0,  11.0],
            [15.0, 14.0, 13.0, 12.0]]

input: Tensor of rank r >= 2.
input: Tensor specifying lengths of the sequences in a batch. It has shape `[batch_size]`.
output: Tensor with same shape of input.
*/

//ReverseSequence
//INPUTS:                   input_input, sequence_lens_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      batch_axis, time_axis
//OPTIONAL_PARAMETERS_TYPE: int, int

//class stuff
namespace backend {   

    class ReverseSequence : public Layer {
        typedef struct {
            int batch_axis; int time_axis;
			
            Shape_t input_input; Shape_t sequence_lens_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        int batch_axis; int time_axis;
        std::string input_input; std::string sequence_lens_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ReverseSequence();
    
        void forward() { program->run(); }
        
        void init( int _batch_axis,  int _time_axis); 
        void bind(std::string _input_input, std::string _sequence_lens_input, std::string _Y_output); 

        ~ReverseSequence() {}
    };

    
    void init_layer_ReverseSequence(py::module& m) {
        // py::class_(m, "ReverseSequence");
    }
    

}


#endif

