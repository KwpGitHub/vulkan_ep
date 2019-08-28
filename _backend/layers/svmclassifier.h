#ifndef SVMCLASSIFIER_H
#define SVMCLASSIFIER_H 

#include "../layer.h"

/*

    Support Vector Machine classifier

input: Data to be classified.
output: Classification outputs (one class per example).
output: Class scores (one per class per example), if prob_a and prob_b are provided they are probabilities for each class, otherwise they are raw scores.
*/

//SVMClassifier
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o, Z_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      classlabels_ints, classlabels_strings, coefficients, kernel_params, kernel_type, post_transform, prob_a, prob_b, rho, support_vectors, vectors_per_class
//OPTIONAL_PARAMETERS_TYPE: std::vector<int>, std::vector<std::string>, std::vector<float>, std::vector<float>, std::string, std::string, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<int>


//class stuff
namespace layers {   

    class SVMClassifier : public backend::Layer {
        typedef struct {
            uint32_t size; float a;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<int> classlabels_ints; std::vector<std::string> classlabels_strings; std::vector<float> coefficients; std::vector<float> kernel_params; std::string kernel_type; std::string post_transform; std::vector<float> prob_a; std::vector<float> prob_b; std::vector<float> rho; std::vector<float> support_vectors; std::vector<int> vectors_per_class;
        std::string X_i;
        
        std::string Y_o; std::string Z_o;
        

        binding_descriptor   binding;
       

    public:
        SVMClassifier(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<int> _classlabels_ints,  std::vector<std::string> _classlabels_strings,  std::vector<float> _coefficients,  std::vector<float> _kernel_params,  std::string _kernel_type,  std::string _post_transform,  std::vector<float> _prob_a,  std::vector<float> _prob_b,  std::vector<float> _rho,  std::vector<float> _support_vectors,  std::vector<int> _vectors_per_class); 
        virtual void bind(std::string _X_i, std::string _Y_o, std::string _Z_o); 
        virtual void build();

        ~SVMClassifier() {}
    };
   
}
#endif

