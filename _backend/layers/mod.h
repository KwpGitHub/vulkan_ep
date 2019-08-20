#ifndef MOD_H
#define MOD_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

  Performs element-wise binary modulus (with Numpy-style broadcasting support). 
    The sign of the remainder is the same as that of the Divisor.
  
    Mod operator can also behave like C fmod() or numpy.fmod. In this case, the sign of the remainder however, will be the same as the Dividend 
    (in contrast to integer mod). To force a behavior like numpy.fmod() an 'fmod' Attribute is provided.
    This attribute is set to 0 by default causing the behavior to be like integer mod. 
    Setting this attribute to 1 causes the remainder to be calculated similar to that of numpy.fmod().

    If the input type is floating point, then `fmod` attribute must be set to 1.
  
    In case of dividend being zero, the results will be platform dependent.

  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

input: Dividend tensor
input: Divisor tensor
output: Remainder tensor
*/

//Mod
//INPUTS:                   A_input, B_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   C_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      fmod
//OPTIONAL_PARAMETERS_TYPE: int

//class stuff
namespace backend {   

    class Mod : public Layer {
        typedef struct {
            int fmod;
			
            Shape_t A_input; Shape_t B_input;
            
            Shape_t C_output;
            
        } binding_descriptor;

        int fmod;
        std::string A_input; std::string B_input;
        
        std::string C_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Mod(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( int _fmod); 
        void bind(std::string _A_input, std::string _B_input, std::string _C_output); 

        ~Mod() {}
    };

}

#endif

