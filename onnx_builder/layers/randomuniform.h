#include <vector>
namespace backend {
    class RandomUniform {
    public:
        RandomUniform ();
        ~RandomUniform();
    private:

		int dtype;
		float high;
		float low;
		float seed;
		std::vector<int> shape;
    };
}
