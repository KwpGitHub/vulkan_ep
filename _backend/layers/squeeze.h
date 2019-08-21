#ifndef SQUEEZE_H
#define SQUEEZE_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Remove single-dimensional entries from the shape of a tensor.
Takes a  parameter `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.

input: Tensors with at least max(dims) dimensions.
output: Reshaped tensor with same data as input.
*/

//Squeeze
//INPUTS:                   data_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   squeezed_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axes
//OPTIONAL_PARAMETERS_TYPE: Shape_t


//class stuff
namespace backend {   

    class Squeeze : public Layer {
        typedef struct {
            Shape_t axes;
			
            Shape_t data_i;
            
            Shape_t squeezed_o;
            
        } binding_descriptor;

        Shape_t axes;
        std::string data_i;
        
        std::string squeezed_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Squeeze(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( Shape_t _axes); 
        virtual void bind(std::string _data_i, std::string _squeezed_o); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/squeeze.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[squeezed_o]->data());
        }

        ~Squeeze() {}
    };
   
}
#endif

