#include <vector>
namespace backend {
    class RandomNormalLike {
    public:
        RandomNormalLike ();
        ~RandomNormalLike();
    private:

		int dtype;
		float mean;
		float scale;
		float seed;
    };
}
