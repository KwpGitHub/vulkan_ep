#ifndef COMPRESS_H
#define COMPRESS_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
    In case axis is not provided, input is flattened before elements are selected.
    Compress behaves like numpy.compress: https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html
    
input: Tensor of rank r >= 1.
input: Rank 1 tensor of booleans to indicate which slices or data elements to be selected. Its length can be less than the input length alone the axis or the flattened input size if axis is not specified. In such cases data slices or elements exceeding the condition length are discarded.
output: Tensor of rank r if axis is specified. Otherwise output is a Tensor of rank 1.
//*/
//Compress
//INPUTS:                   input_input, condition_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis
//OPTIONAL_PARAMETERS_TYPE: int

namespace py = pybind11;

//class stuff
namespace backend {   

    class Compress : public Layer {
        typedef struct {
            int axis;
			
            Shape_t input_input; Shape_t condition_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        int axis;
        std::string input_input; std::string condition_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Compress(std::string n, int axis);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string input_input, std::string condition_input, std::string output_output); 

        ~Compress() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Compress::Compress(std::string n, int axis) : Layer(n) { }
       
    vuh::Device* Compress::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Compress::init() {      
    
		binding.input_input = tensor_dict[input_input]->shape();
  		binding.condition_input = tensor_dict[condition_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.axis = axis;
 
    }
    
    void Compress::call(std::string input_input, std::string condition_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/compress.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[condition_input]->data(), *tensor_dict[output_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Compress, Layer>(m, "Compress")
            .def(py::init<std::string, int> ())
            .def("forward", &Compress::forward)
            .def("init", &Compress::init)
            .def("call", (void (Compress::*) (std::string, std::string, std::string)) &Compress::call);
    }
}

#endif

/* PYTHON STUFF

*/

