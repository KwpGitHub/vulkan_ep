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
//*/
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
			
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        int cast_to; int map_form; int max_map;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        CastMap(std::string n, int cast_to, int map_form, int max_map);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string Y_output); 

        ~CastMap() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    CastMap::CastMap(std::string n, int cast_to, int map_form, int max_map) : Layer(n) { }
       
    vuh::Device* CastMap::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void CastMap::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.cast_to = cast_to;
  		binding.map_form = map_form;
  		binding.max_map = max_map;
 
    }
    
    void CastMap::call(std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/castmap.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<CastMap, Layer>(m, "CastMap")
            .def(py::init<std::string, int, int, int> ())
            .def("forward", &CastMap::forward)
            .def("init", &CastMap::init)
            .def("call", (void (CastMap::*) (std::string, std::string)) &CastMap::call);
    }
}

#endif

/* PYTHON STUFF

*/

