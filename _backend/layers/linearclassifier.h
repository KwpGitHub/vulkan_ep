#ifndef LINEARCLASSIFIER_H
#define LINEARCLASSIFIER_H //LinearClassifier

//INPUTS:                   X
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y, Z
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               coefficients
//PARAMETER_TYPES:          Tensor*
//OPTIONAL_PARAMETERS:      classlabels_ints, classlabels_strings, intercepts, multi_class, post_transform
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*, Tensor*, int, int



namespace backend {
    class LinearClassifier : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            Shape_t classlabels_ints; int multi_class; int post_transform;
			Shape_t coefficients; Shape_t classlabels_strings; Shape_t intercepts;
            //input
            Shape_t X;
            
            //output
            Shape_t Y; Shape_t Z;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        LinearClassifier(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        Tensor* coefficients; Shape_t classlabels_ints; Tensor* classlabels_strings; Tensor* intercepts; int multi_class; int post_transform;
		Shape_t coefficients; Shape_t classlabels_strings; Shape_t intercepts;
        //input
        std::string X;
        
        //output
        std::string Y; std::string Z;
        
        //std::vector<uint32_t> output_shape();
   
        ~LinearClassifier(){}
    };
}


namespace backend {    
    LinearClassifier::LinearClassifier(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/linearclassifier.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* LinearClassifier::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
