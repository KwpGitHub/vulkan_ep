#ifndef CASTMAP_H
#define CASTMAP_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Converts a map to a tensor.<br>The map key must be an int64 and the values will be ordered
    in ascending order based on this key.<br>The operator supports dense packing or sparse packing.
    If using sparse packing, the key cannot exceed the max_map-1 value.

input: The input map that is to be cast to a tensor
output: A tensor representing the same data as the input map, ordered by their keys

*/
//CastMap
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      cast_to, map_form, max_map
//OPTIONAL_PARAMETERS_TYPE: int, int, int

namespace py = pybind11;

//class stuff
namespace backend {   

    class CastMap : public Layer {
        typedef struct {    
            int cast_to; int map_form; int max_map;
        } parameter_descriptor;  

        typedef struct {
            Tensor* X_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* Y_output;
            
        } output_descriptor;

        typedef struct {
            int cast_to; int map_form; int max_map;
		
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        CastMap(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~CastMap() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    CastMap::CastMap(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/castmap.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* CastMap::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void CastMap::init() {
		binding.X_input = input.X_input->shape();
 
		binding.Y_output = output.Y_output->shape();
 
		binding.cast_to = parameters.cast_to;
  		binding.map_form = parameters.map_form;
  		binding.max_map = parameters.max_map;
 
        program->bind(binding, *input.X_input->data(), *output.Y_output->data());
    }
    
    void CastMap::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<CastMap, Layer>(m, "CastMap")
            .def("forward", &CastMap::forward);    
    }
}*/

#endif
