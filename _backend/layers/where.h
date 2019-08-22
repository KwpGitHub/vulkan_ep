#ifndef WHERE_H
#define WHERE_H 

#include "../layer.h"

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
namespace layers {   

    class Where : public backend::Layer {
        typedef struct {          
            backend::Shape_t condition_i; backend::Shape_t X_i; backend::Shape_t Y_i;
            
            backend::Shape_t output_o;
            
        } binding_descriptor;
        using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;

        
        std::string condition_i; std::string X_i; std::string Y_i;
        
        std::string output_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Where(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init(); 
        virtual void bind(std::string _condition_i, std::string _X_i, std::string _Y_i, std::string _output_o); 
        virtual void build();

        ~Where() {}
    };
   
}
#endif

