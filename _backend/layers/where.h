#ifndef WHERE_H
#define WHERE_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

    Return elements, either from X or Y, depending on condition
    (with Numpy-style broadcasting support).
    Where behaves like numpy.where with three parameters:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html

input: When True (nonzero), yield X, otherwise yield Y
input: values selected at indices where condition is True
input: values selected at indices where condition is False
output: Tensor of shape equal to the broadcasted shape of condition, X, and Y.
*/

//Where
//INPUTS:                   condition_i, X_i, Y_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Where : public Layer {
        typedef struct {
            
			
            Shape_t condition_i; Shape_t X_i; Shape_t Y_i;
            
            Shape_t output_o;
            
        } binding_descriptor;

        
        std::string condition_i; std::string X_i; std::string Y_i;
        
        std::string output_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Where(const std::string& name);
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _condition_i, std::string _X_i, std::string _Y_i, std::string _output_o); 

        ~Where() {}
    };

}

#endif

