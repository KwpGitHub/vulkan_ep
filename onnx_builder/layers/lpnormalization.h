#include <vector>
namespace backend {
    class LpNormalization {
    public:
        LpNormalization ();
        ~LpNormalization();
    private:

		int axis;
		int p;
    };
}
