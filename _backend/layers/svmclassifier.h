#ifndef SVMCLASSIFIER_H
#define SVMCLASSIFIER_H //SVMClassifier

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_input_o, Z_input_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      classlabels_ints, classlabels_strings, coefficients, kernel_params, kernel_type, post_transform, prob_a, prob_b, rho, support_vectors, vectors_per_class
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*, Tensor*, Tensor*, int, int, Tensor*, Tensor*, Tensor*, Tensor*, Shape_t



namespace backend {
    class SVMClassifier : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            Shape_t classlabels_ints; int kernel_type; int post_transform; Shape_t vectors_per_class;
			Shape_t classlabels_strings; Shape_t coefficients; Shape_t kernel_params; Shape_t prob_a; Shape_t prob_b; Shape_t rho; Shape_t support_vectors;
            //input
            Shape_t X_input;
            
            //output
            Shape_t Y_input_o; Shape_t Z_input_o;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        SVMClassifier(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        Shape_t classlabels_ints; Tensor* classlabels_strings; Tensor* coefficients; Tensor* kernel_params; int kernel_type; int post_transform; Tensor* prob_a; Tensor* prob_b; Tensor* rho; Tensor* support_vectors; Shape_t vectors_per_class;
		Shape_t classlabels_strings_t; Shape_t coefficients_t; Shape_t kernel_params_t; Shape_t prob_a_t; Shape_t prob_b_t; Shape_t rho_t; Shape_t support_vectors_t;
        //input
        std::string X_input;
        
        //output
        std::string Y_input_o; std::string Z_input_o;
        
        //std::vector<uint32_t> output_shape();
   
        ~SVMClassifier(){}
    };
}


namespace backend {    
    SVMClassifier::SVMClassifier(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/svmclassifier.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({classlabels_ints, kernel_type, post_transform, vectors_per_class, classlabels_strings_t, coefficients_t, kernel_params_t, prob_a_t, prob_b_t, rho_t, support_vectors_t}, 
                            tensor_dict[X_input],
                            tensor_dict[Y_input_o], tensor_dict[Z_input_o] );
    }

    vuh::Device* SVMClassifier::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
