#ifndef CASTMAP_H
#define CASTMAP_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

    Converts a map to a tensor.<br>The map key must be an int64 and the values will be ordered
    in ascending order based on this key.<br>The operator supports dense packing or sparse packing.
    If using sparse packing, the key cannot exceed the max_map-1 value.

input: The input map that is to be cast to a tensor
output: A tensor representing the same data as the input map, ordered by their keys
*/

//CastMap
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      cast_to, map_form, max_map
//OPTIONAL_PARAMETERS_TYPE: int, int, int


//class stuff
namespace backend {   

    class CastMap : public Layer {
        typedef struct {
            int cast_to; int map_form; int max_map;
			
            Shape_t X_i;
            
            Shape_t Y_o;
            
        } binding_descriptor;

        int cast_to; int map_form; int max_map;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        CastMap(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( int _cast_to,  int _map_form,  int _max_map); 
        virtual void bind(std::string _X_i, std::string _Y_o); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/castmap.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data());
        }

        ~CastMap() {}
    };
   
}
#endif

