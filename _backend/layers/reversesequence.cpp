#include "ReverseSequence.h"

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

    py::module m("_backend.nn", "nn MOD");

//python stuff


