#ifndef LRN_H
#define LRN_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Local Response Normalization proposed in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
It normalizes over local input regions.
The local region is defined across the channels. For an element X[n, c, d1, ..., dk] in a tensor
of shape (N x C x D1 x D2, ..., Dk), its region is
{X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}.

square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2),
where max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2)).

Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta

input: Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size. Optionally, if dimension denotation is in effect, the operation expects the input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].
output: Output tensor, which has the shape and type as input tensor
//*/
//LRN
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               size
//PARAMETER_TYPES:          int
//OPTIONAL_PARAMETERS:      alpha, beta, bias
//OPTIONAL_PARAMETERS_TYPE: float, float, float

namespace py = pybind11;

//class stuff
namespace backend {   

    class LRN : public Layer {
        typedef struct {
            int size; float alpha; float beta; float bias;
			
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        int size; float alpha; float beta; float bias;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        LRN(std::string n, int size, float alpha, float beta, float bias);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string Y_output); 

        ~LRN() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    LRN::LRN(std::string n, int size, float alpha, float beta, float bias) : Layer(n) { }
       
    vuh::Device* LRN::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void LRN::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.size = size;
  		binding.alpha = alpha;
  		binding.beta = beta;
  		binding.bias = bias;
 
    }
    
    void LRN::call(std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/lrn.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<LRN, Layer>(m, "LRN")
            .def(py::init<std::string, int, float, float, float> ())
            .def("forward", &LRN::forward)
            .def("init", &LRN::init)
            .def("call", (void (LRN::*) (std::string, std::string)) &LRN::call);
    }
}

#endif

/* PYTHON STUFF

*/

