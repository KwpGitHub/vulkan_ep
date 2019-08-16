#include "../layer.h"
#ifndef WHERE_H
#define WHERE_H 
/*

    Return elements, either from X or Y, depending on condition
    (with Numpy-style broadcasting support).
    Where behaves like numpy.where with three parameters:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html

input: When True (nonzero), yield X, otherwise yield Y
input: values selected at indices where condition is True
input: values selected at indices where condition is False
output: Tensor of shape equal to the broadcasted shape of condition, X, and Y.
//*/
//Where
//INPUTS:                   condition_input, X_input, Y_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Where : public Layer {
        typedef struct {
            
			
            Shape_t condition_input; Shape_t X_input; Shape_t Y_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        
        std::string condition_input; std::string X_input; std::string Y_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Where(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _condition_input, std::string _X_input, std::string _Y_input, std::string _output_output); 

        ~Where() {}

    };
    
}

#endif

