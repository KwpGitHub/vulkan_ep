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

*/
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
        } parameter_descriptor;  

        typedef struct {
            Tensor* data_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* squeezed_output;
            
        } output_descriptor;

        typedef struct {
            Shape_t axes;
		
            Shape_t data_input;
            
            Shape_t squeezed_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Squeeze(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~Squeeze() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Squeeze::Squeeze(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/squeeze.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* Squeeze::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Squeeze::init() {
		binding.data_input = input.data_input->shape();
 
		binding.squeezed_output = output.squeezed_output->shape();
 
		binding.axes = parameters.axes;
 
        program->bind(binding, *input.data_input->data(), *output.squeezed_output->data());
    }
    
    void Squeeze::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Squeeze, Layer>(m, "Squeeze")
            .def("forward", &Squeeze::forward);    
    }
}*/

#endif
