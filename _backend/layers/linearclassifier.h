#ifndef LINEARCLASSIFIER_H
#define LINEARCLASSIFIER_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Linear classifier

input: Data to be classified.
output: Classification outputs (one class per example).
output: Classification scores ([N,E] - one score for each class and example

*/
//LinearClassifier
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output, Z_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               coefficients
//PARAMETER_TYPES:          Tensor*
//OPTIONAL_PARAMETERS:      classlabels_ints, classlabels_strings, intercepts, multi_class, post_transform
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*, Tensor*, int, int

namespace py = pybind11;

//class stuff
namespace backend {   

    class LinearClassifier : public Layer {
        typedef struct {    
            Tensor* coefficients; Shape_t classlabels_ints; Tensor* classlabels_strings; Tensor* intercepts; int multi_class; int post_transform;
        } parameter_descriptor;  

        typedef struct {
            Tensor* X_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* Y_output; Tensor* Z_output;
            
        } output_descriptor;

        typedef struct {
            Shape_t classlabels_ints; int multi_class; int post_transform;
		Shape_t coefficients; Shape_t classlabels_strings; Shape_t intercepts;
            Shape_t X_input;
            
            Shape_t Y_output; Shape_t Z_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        LinearClassifier(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~LinearClassifier() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    LinearClassifier::LinearClassifier(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/linearclassifier.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* LinearClassifier::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void LinearClassifier::init() {
		binding.X_input = input.X_input->shape();
 
		binding.Y_output = output.Y_output->shape();
  		binding.Z_output = output.Z_output->shape();
 
		binding.classlabels_ints = parameters.classlabels_ints;
  		binding.multi_class = parameters.multi_class;
  		binding.post_transform = parameters.post_transform;
  		binding.coefficients = parameters.coefficients->shape();
  		binding.classlabels_strings = parameters.classlabels_strings->shape();
  		binding.intercepts = parameters.intercepts->shape();
 
        program->bind(binding, *parameters.coefficients->data(), *parameters.classlabels_strings->data(), *parameters.intercepts->data(), *input.X_input->data(), *output.Y_output->data(), *output.Z_output->data());
    }
    
    void LinearClassifier::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<LinearClassifier, Layer>(m, "LinearClassifier")
            .def("forward", &LinearClassifier::forward);    
    }
}*/

#endif
