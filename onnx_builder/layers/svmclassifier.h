#include <vector>
namespace backend {
    class SVMClassifier {
    public:
        SVMClassifier ();
        ~SVMClassifier();
    private:

		std::vector<int> classlabels_ints;
		std::vector<string> classlabels_strings;
		std::vector<float> coefficients;
		std::vector<float> kernel_params;
		std::string kernel_type;
		std::string post_transform;
		std::vector<float> prob_a;
		std::vector<float> prob_b;
		std::vector<float> rho;
		std::vector<float> support_vectors;
		std::vector<int> vectors_per_class;
    };
}
