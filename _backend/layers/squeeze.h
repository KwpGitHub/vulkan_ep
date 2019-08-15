#ifndef SQUEEZE_H
#define SQUEEZE_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Remove single-dimensional entries from the shape of a tensor.
Takes a  parameter `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.

input: Tensors with at least max(dims) dimensions.
output: Reshaped tensor with same data as input.
//*/
//Squeeze
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   squeezed_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axes
//OPTIONAL_PARAMETERS_TYPE: Shape_t

namespace py = pybind11;

//class stuff
namespace backend {   

    class Squeeze : public Layer {
        typedef struct {
            Shape_t axes;
			
            Shape_t data_input;
            
            Shape_t squeezed_output;
            
        } binding_descriptor;

        Shape_t axes;
        std::string data_input;
        
        std::string squeezed_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Squeeze(std::string n, Shape_t axes);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string data_input, std::string squeezed_output); 

        ~Squeeze() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Squeeze::Squeeze(std::string n, Shape_t axes) : Layer(n) { }
       
    vuh::Device* Squeeze::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Squeeze::init() {      
    
		binding.data_input = tensor_dict[data_input]->shape();
 
		binding.squeezed_output = tensor_dict[squeezed_output]->shape();
 
		binding.axes = axes;
 
    }
    
    void Squeeze::call(std::string data_input, std::string squeezed_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/squeeze.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[squeezed_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Squeeze, Layer>(m, "Squeeze")
            .def(py::init<std::string, Shape_t> ())
            .def("forward", &Squeeze::forward)
            .def("init", &Squeeze::init)
            .def("call", (void (Squeeze::*) (std::string, std::string)) &Squeeze::call);
    }
}

#endif

/* PYTHON STUFF

*/

