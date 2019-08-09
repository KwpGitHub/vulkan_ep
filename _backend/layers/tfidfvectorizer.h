#ifndef TFIDFVECTORIZER_H
#define TFIDFVECTORIZER_H //TfIdfVectorizer

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_input_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               max_gram_length, max_skip_count, min_gram_length, mode, ngram_counts, ngram_indexes
//PARAMETER_TYPES:          int, int, int, int, Shape_t, Shape_t
//OPTIONAL_PARAMETERS:      pool_int64s, pool_strings, weights
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*, Tensor*



namespace backend {
    class TfIdfVectorizer : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int max_gram_length; int max_skip_count; int min_gram_length; int mode; Shape_t ngram_counts; Shape_t ngram_indexes; Shape_t pool_int64s;
			Shape_t pool_strings; Shape_t weights;
            //input
            Shape_t X_input;
            
            //output
            Shape_t Y_input_o;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        TfIdfVectorizer(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        int max_gram_length; int max_skip_count; int min_gram_length; int mode; Shape_t ngram_counts; Shape_t ngram_indexes; Shape_t pool_int64s; Tensor* pool_strings; Tensor* weights;
		Shape_t pool_strings_t; Shape_t weights_t;
        //input
        std::string X_input;
        
        //output
        std::string Y_input_o;
        
        //std::vector<uint32_t> output_shape();
   
        ~TfIdfVectorizer(){}
    };
}


namespace backend {    
    TfIdfVectorizer::TfIdfVectorizer(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/tfidfvectorizer.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({max_gram_length, max_skip_count, min_gram_length, mode, ngram_counts, ngram_indexes, pool_int64s, pool_strings_t, weights_t}, 
                            tensor_dict[X_input],
                            tensor_dict[Y_input_o] );
    }

    vuh::Device* TfIdfVectorizer::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
