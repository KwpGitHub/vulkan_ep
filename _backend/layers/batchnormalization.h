#ifndef BATCHNORMALIZATION_H
#define BATCHNORMALIZATION_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)

For previous (depreciated) non-spatial cases, implementors are suggested
to flatten the input shape to (N x C*D1*D2 ..*Dn) before a BatchNormalization Op.
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

input: Input data tensor from the previous operator; dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size, C is the number of channels. Statistics are computed for every channel of C over N and D1 to Dn dimensions. For image data, input dimensions become (N x C x H x W). The op also accepts single dimension input of size N in which case C is assumed to be 1
input: Scale tensor of shape (C).
input: Bias tensor of shape (C).
input: running (training) or estimated (testing) mean tensor of shape (C).
input: running (training) or estimated (testing) variance tensor of shape (C).
output: The output tensor of the same shape as X
output: The running mean after the BatchNormalization operator.
output: The running variance after the BatchNormalization operator.
output: Saved mean used during training to speed up gradient computation.
output: Saved variance used during training to speed up gradient computation.
//*/
//BatchNormalization
//INPUTS:                   X_input, scale_input, B_input, mean_input, var_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         mean_output_opt, var_output_opt, saved_mean_output_opt, saved_var_output_opt
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      epsilon, momentum
//OPTIONAL_PARAMETERS_TYPE: float, float

namespace py = pybind11;

//class stuff
namespace backend {   

    class BatchNormalization : public Layer {
        typedef struct {
            float epsilon; float momentum;
			
            Shape_t X_input; Shape_t scale_input; Shape_t B_input; Shape_t mean_input; Shape_t var_input;
            
            Shape_t Y_output;
            Shape_t mean_output_opt; Shape_t var_output_opt; Shape_t saved_mean_output_opt; Shape_t saved_var_output_opt;
        } binding_descriptor;

        float epsilon; float momentum;
        std::string X_input; std::string scale_input; std::string B_input; std::string mean_input; std::string var_input;
        
        std::string Y_output;
        std::string mean_output_opt; std::string var_output_opt; std::string saved_mean_output_opt; std::string saved_var_output_opt;

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        BatchNormalization(std::string n, float epsilon, float momentum);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string scale_input, std::string B_input, std::string mean_input, std::string var_input, std::string Y_output, std::string mean_output_opt, std::string var_output_opt, std::string saved_mean_output_opt, std::string saved_var_output_opt); 

        ~BatchNormalization() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    BatchNormalization::BatchNormalization(std::string n, float epsilon, float momentum) : Layer(n) { }
       
    vuh::Device* BatchNormalization::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void BatchNormalization::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.scale_input = tensor_dict[scale_input]->shape();
  		binding.B_input = tensor_dict[B_input]->shape();
  		binding.mean_input = tensor_dict[mean_input]->shape();
  		binding.var_input = tensor_dict[var_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
  		binding.mean_output_opt = tensor_dict[mean_output_opt]->shape();
  		binding.var_output_opt = tensor_dict[var_output_opt]->shape();
  		binding.saved_mean_output_opt = tensor_dict[saved_mean_output_opt]->shape();
  		binding.saved_var_output_opt = tensor_dict[saved_var_output_opt]->shape();
 
		binding.epsilon = epsilon;
  		binding.momentum = momentum;
 
    }
    
    void BatchNormalization::call(std::string X_input, std::string scale_input, std::string B_input, std::string mean_input, std::string var_input, std::string Y_output, std::string mean_output_opt, std::string var_output_opt, std::string saved_mean_output_opt, std::string saved_var_output_opt){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/batchnormalization.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[scale_input]->data(), *tensor_dict[B_input]->data(), *tensor_dict[mean_input]->data(), *tensor_dict[var_input]->data(), *tensor_dict[Y_output]->data(), *tensor_dict[mean_output_opt]->data(), *tensor_dict[var_output_opt]->data(), *tensor_dict[saved_mean_output_opt]->data(), *tensor_dict[saved_var_output_opt]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<BatchNormalization, Layer>(m, "BatchNormalization")
            .def(py::init<std::string, float, float> ())
            .def("forward", &BatchNormalization::forward)
            .def("init", &BatchNormalization::init)
            .def("call", (void (BatchNormalization::*) (std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string)) &BatchNormalization::call);
    }
}

#endif

/* PYTHON STUFF

*/

