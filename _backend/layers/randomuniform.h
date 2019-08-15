#ifndef RANDOMUNIFORM_H
#define RANDOMUNIFORM_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Generate a tensor with random values drawn from a uniform distribution. The shape
of the tensor is specified by the `shape` argument and the range by `low` and `high`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.


output: Output tensor of random values drawn from uniform distribution

*/
//RandomUniform
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      dtype, high, low, seed
//OPTIONAL_PARAMETERS_TYPE: int, float, float, float

namespace py = pybind11;

//class stuff
namespace backend {   

    class RandomUniform : public Layer {
        typedef struct {    
            Shape_t shape; int dtype; float high; float low; float seed;
        } parameter_descriptor;  

        typedef struct {
            
            
        } input_desriptor;

        typedef struct {
            Tensor* output_output;
            
        } output_descriptor;

        typedef struct {
            Shape_t shape; int dtype; float high; float low; float seed;
		
            
            
            Shape_t output_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        RandomUniform(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~RandomUniform() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    RandomUniform::RandomUniform(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/randomuniform.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* RandomUniform::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void RandomUniform::init() {

		binding.output_output = output.output_output->shape();
 
		binding.shape = parameters.shape;
  		binding.dtype = parameters.dtype;
  		binding.high = parameters.high;
  		binding.low = parameters.low;
  		binding.seed = parameters.seed;
 
        program->bind(binding, *output.output_output->data());
    }
    
    void RandomUniform::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<RandomUniform, Layer>(m, "RandomUniform")
            .def("forward", &RandomUniform::forward);    
    }
}*/

#endif
