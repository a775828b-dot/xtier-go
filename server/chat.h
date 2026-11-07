// The checkpoint's chat template rendered with minja, the way vLLM renders it with transformers + jinja2: tool
// call arguments in the history turned into objects, content parts joined into a string, no polyfills (the
// template is applied as written). tests/template_check.py compares the output with Python's character for
// character.
#pragma once

#include <memory>
#include <string>

#include "xtg/json.h"

namespace minja {
class chat_template;
}

namespace xtg {

class ChatTemplate {
 public:
  explicit ChatTemplate(const std::string& model_dir);
  ~ChatTemplate();

  std::string render(const json& messages, const json& tools, bool add_generation_prompt, const json& kwargs) const;

  // What vLLM does to an OpenAI request's messages before rendering.
  static json prepare_messages(const json& messages);

  const std::string& bos_token() const { return bos_; }
  const std::string& eos_token() const { return eos_; }

 private:
  std::unique_ptr<minja::chat_template> tmpl_;
  std::string bos_, eos_;
};

}  // namespace xtg
