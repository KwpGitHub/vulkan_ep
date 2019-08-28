#ifndef CATEGORYMAPPER_H
#define CATEGORYMAPPER_H 

#include "../layer.h"

/*

    Converts strings to integers and vice versa.<br>
    Two sequences of equal length are used to map between integers and strings,
    with strings and integers at the same index detailing the mapping.<br>
    Each operator converts either integers to strings or strings to integers, depending 
    on which default value attribute is provided. Only one default value attribute
    should be defined.<br>
    If the string default value is set, it will convert integers to strings.
    If the int default value is set, it will convert strings to integers.

input: Input data
output: Output data. If strings are input, the output values are integers, and vice versa.
*/

//CategoryMapper
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      cats_int64s, cats_strings, default_int64, default_string
//OPTIONAL_PARAMETERS_TYPE: std::vector<int>, std::vector<std::string>, int, std::string


//class stuff
namespace layers {   

    class CategoryMapper : public backend::Layer {
        typedef struct {
            uint32_t size; float a;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<int> cats_int64s; std::vector<std::string> cats_strings; int default_int64; std::string default_string;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;
       

    public:
        CategoryMapper(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<int> _cats_int64s,  std::vector<std::string> _cats_strings,  int _default_int64,  std::string _default_string); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~CategoryMapper() {}
    };
   
}
#endif

