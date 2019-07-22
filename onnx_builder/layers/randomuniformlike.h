#include <vector>
namespace backend {
    class RandomUniformLike {
    public:
        RandomUniformLike ();
        ~RandomUniformLike();
    private:

		int dtype;
		float high;
		float low;
		float seed;
    };
}
