#include <vector>
namespace backend {
    class MaxRoiPool {
    public:
        MaxRoiPool ();
        ~MaxRoiPool();
    private:

		std::vector<int> pooled_shape;
		float spatial_scale;
    };
}
