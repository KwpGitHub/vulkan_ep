#ifndef RANDOMNORMALLIKE_H
#define RANDOMNORMALLIKE_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Generate a tensor with random values drawn from a normal distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the normal distribution are specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message, and be valid as an output type.

input: Input tensor to copy shape and optionally type information from.
output: Output tensor of random values drawn from normal distribution

*/
//RandomNormalLike
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      dtype, mean, scale, seed
//OPTIONAL_PARAMETERS_TYPE: int, float, float, float

namespace py = pybind11;

//class stuff
namespace backend {   

    class RandomNormalLike : public Layer {
        typedef struct {    
            int dtype; float mean; float scale; float seed;
        } parameter_descriptor;  

        typedef struct {
            Tensor* input_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* output_output;
            
        } output_descriptor;

        typedef struct {
            int dtype; float mean; float scale; float seed;
		
            Shape_t input_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        RandomNormalLike(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~RandomNormalLike() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    RandomNormalLike::RandomNormalLike(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/randomnormallike.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* RandomNormalLike::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void RandomNormalLike::init() {
		binding.input_input = input.input_input->shape();
 
		binding.output_output = output.output_output->shape();
 
		binding.dtype = parameters.dtype;
  		binding.mean = parameters.mean;
  		binding.scale = parameters.scale;
  		binding.seed = parameters.seed;
 
        program->bind(binding, *input.input_input->data(), *output.output_output->data());
    }
    
    void RandomNormalLike::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<RandomNormalLike, Layer>(m, "RandomNormalLike")
            .def("forward", &RandomNormalLike::forward);    
    }
}*/

#endif
