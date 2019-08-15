#ifndef RANDOMUNIFORMLIKE_H
#define RANDOMUNIFORMLIKE_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Generate a tensor with random values drawn from a uniform distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the uniform distribution are specified by `low` and `high`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.

input: Input tensor to copy shape and optionally type information from.
output: Output tensor of random values drawn from uniform distribution
//*/
//RandomUniformLike
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      dtype, high, low, seed
//OPTIONAL_PARAMETERS_TYPE: int, float, float, float

namespace py = pybind11;

//class stuff
namespace backend {   

    class RandomUniformLike : public Layer {
        typedef struct {
            int dtype; float high; float low; float seed;
			
            Shape_t input_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        int dtype; float high; float low; float seed;
        std::string input_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        RandomUniformLike(std::string n, int dtype, float high, float low, float seed);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string input_input, std::string output_output); 

        ~RandomUniformLike() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    RandomUniformLike::RandomUniformLike(std::string n, int dtype, float high, float low, float seed) : Layer(n) { }
       
    vuh::Device* RandomUniformLike::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void RandomUniformLike::init() {      
    
		binding.input_input = tensor_dict[input_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.dtype = dtype;
  		binding.high = high;
  		binding.low = low;
  		binding.seed = seed;
 
    }
    
    void RandomUniformLike::call(std::string input_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/randomuniformlike.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[output_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<RandomUniformLike, Layer>(m, "RandomUniformLike")
            .def(py::init<std::string, int, float, float, float> ())
            .def("forward", &RandomUniformLike::forward)
            .def("init", &RandomUniformLike::init)
            .def("call", (void (RandomUniformLike::*) (std::string, std::string)) &RandomUniformLike::call);
    }
}

#endif

/* PYTHON STUFF

*/

