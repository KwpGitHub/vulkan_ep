#ifndef LINEARCLASSIFIER_H
#define LINEARCLASSIFIER_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

    Linear classifier

input: Data to be classified.
output: Classification outputs (one class per example).
output: Classification scores ([N,E] - one score for each class and example
*/

//LinearClassifier
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o, Z_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               coefficients
//PARAMETER_TYPES:          Tensor*
//OPTIONAL_PARAMETERS:      classlabels_ints, classlabels_strings, intercepts, multi_class, post_transform
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*, Tensor*, int, int

//class stuff
namespace backend {   

    class LinearClassifier : public Layer {
        typedef struct {
            Shape_t classlabels_ints; int multi_class; int post_transform;
			Shape_t coefficients; Shape_t classlabels_strings; Shape_t intercepts;
            Shape_t X_i;
            
            Shape_t Y_o; Shape_t Z_o;
            
        } binding_descriptor;

        Shape_t classlabels_ints; int multi_class; int post_transform; std::string coefficients; std::string classlabels_strings; std::string intercepts;
        std::string X_i;
        
        std::string Y_o; std::string Z_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        LinearClassifier(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( Shape_t _classlabels_ints,  int _multi_class,  int _post_transform); 
        void bind(std::string _coefficients, std::string _classlabels_strings, std::string _intercepts, std::string _X_i, std::string _Y_o, std::string _Z_o); 

        ~LinearClassifier() {}
    };

}

#endif

