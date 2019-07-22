#include <vector>
namespace backend {
    class RandomNormal {
    public:
        RandomNormal ();
        ~RandomNormal();
    private:

		int dtype;
		float mean;
		float scale;
		float seed;
		std::vector<int> shape;
    };
}
