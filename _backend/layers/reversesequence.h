#ifndef REVERSESEQUENCE_H
#define REVERSESEQUENCE_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
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
//*/
//ReverseSequence
//INPUTS:                   input_input, sequence_lens_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      batch_axis, time_axis
//OPTIONAL_PARAMETERS_TYPE: int, int

namespace py = pybind11;

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
        ReverseSequence(std::string n, int batch_axis, int time_axis);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string input_input, std::string sequence_lens_input, std::string Y_output); 

        ~ReverseSequence() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    ReverseSequence::ReverseSequence(std::string n, int batch_axis, int time_axis) : Layer(n) { }
       
    vuh::Device* ReverseSequence::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void ReverseSequence::init() {      
    
		binding.input_input = tensor_dict[input_input]->shape();
  		binding.sequence_lens_input = tensor_dict[sequence_lens_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.batch_axis = batch_axis;
  		binding.time_axis = time_axis;
 
    }
    
    void ReverseSequence::call(std::string input_input, std::string sequence_lens_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/reversesequence.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[sequence_lens_input]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<ReverseSequence, Layer>(m, "ReverseSequence")
            .def(py::init<std::string, int, int> ())
            .def("forward", &ReverseSequence::forward)
            .def("init", &ReverseSequence::init)
            .def("call", (void (ReverseSequence::*) (std::string, std::string, std::string)) &ReverseSequence::call);
    }
}

#endif

/* PYTHON STUFF

*/

