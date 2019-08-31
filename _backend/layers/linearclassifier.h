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
            int t;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<float> m_coefficients; std::vector<int> m_classlabels_ints; std::vector<std::string> m_classlabels_strings; std::vector<float> m_intercepts; int m_multi_class; std::string m_post_transform;
        std::string m_X_i;
        
        std::string m_Y_o; std::string m_Z_o;
        

        binding_descriptor   binding;
       

    public:
        LinearClassifier(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<float> _coefficients,  std::vector<int> _classlabels_ints,  std::vector<std::string> _classlabels_strings,  std::vector<float> _intercepts,  int _multi_class,  std::string _post_transform); 
        virtual void bind(std::string _X_i, std::string _Y_o, std::string _Z_o); 
        virtual void build();

        ~LinearClassifier() {}
    };
   
}
#endif

