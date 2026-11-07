// JSON that keeps the order of object keys: chat templates render tool schemas with tojson, and the prompt must
// come out the way Python's json.dumps writes the request (vLLM's prompt), not with sorted keys.
#pragma once

#include <nlohmann/json.hpp>

namespace xtg {
using json = nlohmann::ordered_json;
}
