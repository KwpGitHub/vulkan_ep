#ifndef MEANVARIANCENORMALIZATION_H
#define MEANVARIANCENORMALIZATION_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

      A MeanVarianceNormalization Function: Perform mean variance normalization
      on the input tensor X using formula: <br/> ``` (X-EX)/sqrt(E(X-EX)^2) ```

input: Input tensor
output: Output tensor

*/
//MeanVarianceNormalization
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axes
//OPTIONAL_PARAMETERS_TYPE: Shape_t

namespace py = pybind11;

//class stuff
namespace backend {   

    class MeanVarianceNormalization : public Layer {
        typedef struct {    
            Shape_t axes;
        } parameter_descriptor;  

        typedef struct {
            Tensor* X_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* Y_output;
            
        } output_descriptor;

        typedef struct {
            Shape_t axes;
		
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        MeanVarianceNormalization(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~MeanVarianceNormalization() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    MeanVarianceNormalization::MeanVarianceNormalization(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/meanvariancenormalization.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* MeanVarianceNormalization::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void MeanVarianceNormalization::init() {
		binding.X_input = input.X_input->shape();
 
		binding.Y_output = output.Y_output->shape();
 
		binding.axes = parameters.axes;
 
        program->bind(binding, *input.X_input->data(), *output.Y_output->data());
    }
    
    void MeanVarianceNormalization::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<MeanVarianceNormalization, Layer>(m, "MeanVarianceNormalization")
            .def("forward", &MeanVarianceNormalization::forward);    
    }
}*/

#endif
