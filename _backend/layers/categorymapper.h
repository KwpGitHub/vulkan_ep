#include "../layer.h"
#ifndef CATEGORYMAPPER_H
#define CATEGORYMAPPER_H 
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
//*/
//CategoryMapper
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      cats_int64s, cats_strings, default_int64, default_string
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*, int, int

//class stuff
namespace backend {   

    class CategoryMapper : public Layer {
        typedef struct {
            Shape_t cats_int64s; int default_int64; int default_string;
			Shape_t cats_strings;
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        Shape_t cats_int64s; int default_int64; int default_string; std::string cats_strings;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        CategoryMapper(std::string n, Shape_t cats_int64s, int default_int64, int default_string);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string cats_strings, std::string X_input, std::string Y_output); 

        ~CategoryMapper() {}

    };
    
}

#endif

