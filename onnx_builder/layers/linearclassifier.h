#include <vector>
namespace backend {
    class LinearClassifier {
    public:
        LinearClassifier ();
        ~LinearClassifier();
    private:

		std::vector<int> classlabels_ints;
		std::vector<string> classlabels_strings;
		std::vector<float> coefficients;
		std::vector<float> intercepts;
		int multi_class;
		std::string post_transform;
    };
}
