// A Hugging Face tokenizer.json of the byte-level BPE kind (GPT-2 / Tekken style): added tokens matched in the raw
// text first (leftmost, longest), the rest split by the pre-tokenizer regex, each piece mapped to byte-level
// characters and merged by rank. Results must equal the Python tokenizers library token for token
// (tests/tokenizer_check.py); the regex is not interpreted but hand-coded for the pattern this file accepts.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace xtg {

class Tokenizer {
 public:
  explicit Tokenizer(const std::string& tokenizer_json);

  // parse_special: added tokens written in the text become their ids (what a rendered chat prompt needs).
  std::vector<int32_t> encode(const std::string& text, bool parse_special = true) const;
  // The raw bytes of a token: an added token's content, a vocabulary token's bytes.
  const std::string& token_bytes(int32_t id) const;
  std::string decode(const std::vector<int32_t>& ids, bool skip_special) const;
  bool is_special(int32_t id) const { return id >= 0 && id < int32_t(special_.size()) && special_[size_t(id)]; }
  bool is_added(int32_t id) const { return id >= 0 && id < int32_t(added_.size()) && added_[size_t(id)]; }
  // id of an added token or of a vocabulary entry written as raw text; -1 if none
  int32_t find(const std::string& content) const;
  int vocab_size() const { return int(bytes_.size()); }

 private:
  void encode_piece(const std::string& piece, std::vector<int32_t>& out) const;
  void encode_plain(const std::string& text, std::vector<int32_t>& out) const;

  std::unordered_map<std::string, int32_t> vocab_;        // byte-level form -> id
  std::vector<std::string> bytes_;                        // id -> raw bytes
  std::vector<bool> special_, added_;
  std::unordered_map<uint64_t, std::pair<int32_t, int32_t>> merges_;   // (left, right) -> (rank, merged)
  std::vector<std::pair<std::string, int32_t>> added_list_; // longest first
  std::vector<size_t> first_[256];                         // indices into added_list_ by first byte
  std::string byte_char_[256];                            // byte -> its byte-level character (UTF-8)
  bool ignore_merges_ = false;
};

}  // namespace xtg
