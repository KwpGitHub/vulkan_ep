#include <vector>
namespace backend {
    class ReduceMax {
    public:
        ReduceMax ();
        ~ReduceMax();
    private:

		std::vector<int> axes;
		int keepdims;
    };
}
