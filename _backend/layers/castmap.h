#ifndef CASTMAP_H
#define CASTMAP_H 

#include "../layer.h"

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
//OPTIONAL_PARAMETERS_TYPE: std::string, std::string, int


//class stuff
namespace layers {   

    class CastMap : public backend::Layer {
        typedef struct {
            int t;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::string m_cast_to; std::string m_map_form; int m_max_map;
        std::string m_X_i;
        
        std::string m_Y_o;
        

        binding_descriptor   binding;
       

    public:
        CastMap(std::string name);
        
        virtual void forward();        
        virtual void init( std::string _cast_to,  std::string _map_form,  int _max_map); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~CastMap() {}
    };
   
}
#endif

