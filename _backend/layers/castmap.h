#include "../layer.h"
#ifndef CASTMAP_H
#define CASTMAP_H 
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
        CastMap(std::string n);
    
        void forward() { program->run(); }
        
        void init( int _cast_to,  int _map_form,  int _max_map); 
        void bind(std::string _X_input, std::string _Y_output); 

        ~CastMap() {}

    };
    
}

#endif

