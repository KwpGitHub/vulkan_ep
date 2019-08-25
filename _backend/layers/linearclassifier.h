#ifndef LINEARCLASSIFIER_H
#define LINEARCLASSIFIER_H 

#include "../layer.h"

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
//PARAMETER_TYPES:          std::vector<float>
//OPTIONAL_PARAMETERS:      classlabels_ints, classlabels_strings, intercepts, multi_class, post_transform
//OPTIONAL_PARAMETERS_TYPE: std::vector<int>, std::vector<std::string>, std::vector<float>, int, std::string


//class stuff
namespace layers {   

    class LinearClassifier : public backend::Layer {
        typedef struct {          
            backend::Shape_t X_i;
            
            backend::Shape_t Y_o; backend::Shape_t Z_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::vector<float> coefficients; std::vector<int> classlabels_ints; std::vector<std::string> classlabels_strings; std::vector<float> intercepts; int multi_class; std::string post_transform;
        std::string X_i;
        
        std::string Y_o; std::string Z_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        LinearClassifier(std::string name);
        
        void forward() { program->run(); }
        
        virtual void init( std::vector<float> _coefficients,  std::vector<int> _classlabels_ints,  std::vector<std::string> _classlabels_strings,  std::vector<float> _intercepts,  int _multi_class,  std::string _post_transform); 
        virtual void bind(std::string _X_i, std::string _Y_o, std::string _Z_o); 
        virtual void build();

        ~LinearClassifier() {}
    };
   
}
#endif

