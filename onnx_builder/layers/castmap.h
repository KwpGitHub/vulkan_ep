#include <vector>
namespace backend {
    class CastMap {
    public:
        CastMap ();
        ~CastMap();
    private:

		std::string cast_to;
		std::string map_form;
		int max_map;
    };
}
