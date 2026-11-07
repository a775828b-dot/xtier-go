// The model packs built into the engine. A new model family = a directory under packs/ + a line here.
#include "dory/dory.h"
#include "xtg/model.h"

namespace xtg {

void register_builtin_packs() {
  static const bool once = [] {
    dory::register_pack();
    return true;
  }();
  (void)once;
}

}  // namespace xtg
