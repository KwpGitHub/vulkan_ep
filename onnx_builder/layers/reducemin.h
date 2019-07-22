#include <vector>
namespace backend {
    class ReduceMin {
    public:
        ReduceMin ();
        ~ReduceMin();
    private:

		std::vector<int> axes;
		int keepdims;
    };
}
