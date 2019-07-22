#include <vector>
namespace backend {
    class FeatureVectorizer {
    public:
        FeatureVectorizer ();
        ~FeatureVectorizer();
    private:

		std::vector<int> inputdimensions;
    };
}
