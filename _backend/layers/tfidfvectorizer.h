#ifndef TFIDFVECTORIZER_H
#define TFIDFVECTORIZER_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class TfIdfVectorizer : public Layer {
        struct Params{Shape_t X_t; Shape_t Y_t; int max_gram_length_t; int max_skip_count_t; int min_gram_length_t; int mode_t; Shape_t ngram_counts_t; Shape_t ngram_indexes_t; Shape_t pool_int64s_t; int* pool_strings_t;
        };
            
		Tensor* weights;
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string X; std::string Y;
        //parameter 
        Shape_t X_t; Shape_t Y_t; int max_gram_length_t; int max_skip_count_t; int min_gram_length_t; int mode_t; Shape_t ngram_counts_t; Shape_t ngram_indexes_t; Shape_t pool_int64s_t; int* pool_strings_t;

    public:
        TfIdfVectorizer(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            X = i[0];
            Y = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/tfidfvectorizer.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({X_t, Y_t, max_gram_length_t, max_skip_count_t, min_gram_length_t, mode_t, ngram_counts_t, ngram_indexes_t, pool_int64s_t, pool_strings_t }, tensor_dict[Y], tensor_dict[X]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["X"], X_t);
			convert_vec_param(a["Y"], Y_t);
			convert_vec_param(a["max_gram_length"], max_gram_length_t);
			convert_vec_param(a["max_skip_count"], max_skip_count_t);
			convert_vec_param(a["min_gram_length"], min_gram_length_t);
			convert_vec_param(a["mode"], mode_t);
			convert_vec_param(a["ngram_counts"], ngram_counts_t);
			convert_vec_param(a["ngram_indexes"], ngram_indexes_t);
			convert_vec_param(a["pool_int64s"], pool_int64s_t);
			convert_vec_param(a["pool_strings"], pool_strings_t);   
        }

        //Tensor* operator()(const Tensor* t) {            
        //}

		void forward(){
		}

       /* std::vector<uint32_t> output_shape(){
            for(auto t_name : inputs){
                if(tensor_dict.end() == tensor_dict.find(t_name) && layer_dict.end() != layer_dict.find(t_name)){
                    //need to do math
                    return layer_dict[t_name]->output_shape();
                }
                else if (tensor_dict.end() != tensor_dict.find(t_name) && layer_dict.end() == layer_dict.find(t_name)){
                    //need to do math
                    return tensor_dict[t_name]->dims;
                }

            }
            for(auto t_name : outputs){
                if(tensor_dict.end() != tensor_dict.find(t_name) && layer_dict.end() == layer_dict.find(t_name)){
                    return tensor_dict[t_name]->dims;
                }
            }
        }*/

    
        ~TfIdfVectorizer(){}

    };
}

#endif
