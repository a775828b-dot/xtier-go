#include "tokenizer.h"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <queue>
#include <sstream>

#include "xtg/common.h"

namespace xtg {

namespace {

const uint32_t kRuns[][2] = {
#include "unicode_ranges.inc"
};

enum : uint8_t { kUpper = 1, kLower = 2, kOtherL = 4, kMark = 8, kNum = 16, kSpace = 32 };

uint8_t cp_flags(uint32_t cp) {
  size_t lo = 0, hi = sizeof(kRuns) / sizeof(kRuns[0]);
  while (hi - lo > 1) {
    size_t mid = (lo + hi) / 2;
    if (kRuns[mid][0] <= cp)
      lo = mid;
    else
      hi = mid;
  }
  return uint8_t(kRuns[lo][1]);
}

// The pre-tokenizer pattern this file implements (Tekken / o200k style); tokenizer.json must carry exactly it.
const char* kPattern =
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|"
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|"
    "\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

// UTF-8 -> code points with their byte offsets (an invalid byte stands for itself, flags 0)
void decode_utf8(const std::string& s, std::vector<uint32_t>& cps, std::vector<size_t>& offs) {
  cps.clear();
  offs.clear();
  size_t i = 0, n = s.size();
  while (i < n) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    uint32_t cp;
    size_t len;
    if (c < 0x80) {
      cp = c;
      len = 1;
    } else if ((c >> 5) == 6 && i + 1 < n) {
      cp = (uint32_t(c & 0x1f) << 6) | (uint32_t(s[i + 1]) & 0x3f);
      len = 2;
    } else if ((c >> 4) == 14 && i + 2 < n) {
      cp = (uint32_t(c & 0x0f) << 12) | ((uint32_t(s[i + 1]) & 0x3f) << 6) | (uint32_t(s[i + 2]) & 0x3f);
      len = 3;
    } else if ((c >> 3) == 30 && i + 3 < n) {
      cp = (uint32_t(c & 0x07) << 18) | ((uint32_t(s[i + 1]) & 0x3f) << 12) | ((uint32_t(s[i + 2]) & 0x3f) << 6) |
           (uint32_t(s[i + 3]) & 0x3f);
      len = 4;
    } else {
      cp = 0xFFFFFFFF;
      len = 1;
    }
    cps.push_back(cp);
    offs.push_back(i);
    i += len;
  }
  offs.push_back(n);
}

// Split text into the pieces the regex matches, as [begin, end) byte ranges. Every position matches one of the
// alternatives; each is tried in order with the backtracking a regex engine would do.
void split_pieces(const std::string& text, std::vector<std::pair<size_t, size_t>>& pieces) {
  std::vector<uint32_t> cp;
  std::vector<size_t> off;
  decode_utf8(text, cp, off);
  const size_t n = cp.size();
  std::vector<uint8_t> fl(n);
  for (size_t j = 0; j < n; ++j) fl[j] = cp[j] == 0xFFFFFFFF ? 0 : cp_flags(cp[j]);
  auto isL = [&](size_t j) { return (fl[j] & (kUpper | kLower | kOtherL)) != 0; };
  auto isN = [&](size_t j) { return (fl[j] & kNum) != 0; };
  auto isS = [&](size_t j) { return (fl[j] & kSpace) != 0; };
  auto clsA = [&](size_t j) { return (fl[j] & (kUpper | kOtherL | kMark)) != 0; };   // [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]
  auto clsB = [&](size_t j) { return (fl[j] & (kLower | kOtherL | kMark)) != 0; };   // [\p{Ll}\p{Lm}\p{Lo}\p{M}]
  auto crlf = [&](size_t j) { return cp[j] == '\r' || cp[j] == '\n'; };
  auto prefix = [&](size_t j) { return !crlf(j) && !isL(j) && !isN(j); };            // [^\r\n\p{L}\p{N}]
  auto punct = [&](size_t j) { return !isS(j) && !isL(j) && !isN(j); };              // [^\s\p{L}\p{N}]

  auto match = [&](size_t i) -> size_t {
    // 1: prefix? A* B+
    for (int pre = 1; pre >= 0; --pre) {
      if (pre && !prefix(i)) continue;
      size_t s = i + size_t(pre), a = s;
      while (a < n && clsA(a)) ++a;
      for (size_t k = a;; --k) {
        if (k < n && clsB(k)) {
          size_t e = k;
          while (e < n && clsB(e)) ++e;
          return e;
        }
        if (k == s) break;
      }
    }
    // 2: prefix? A+ B*
    for (int pre = 1; pre >= 0; --pre) {
      if (pre && !prefix(i)) continue;
      size_t s = i + size_t(pre), a = s;
      while (a < n && clsA(a)) ++a;
      if (a == s) continue;
      while (a < n && clsB(a)) ++a;
      return a;
    }
    // 3: one number
    if (isN(i)) return i + 1;
    // 4: ' '? punct+ [\r\n/]*
    for (int sp = 1; sp >= 0; --sp) {
      if (sp && cp[i] != ' ') continue;
      size_t s = i + size_t(sp), e = s;
      while (e < n && punct(e)) ++e;
      if (e == s) continue;
      while (e < n && (crlf(e) || cp[e] == '/')) ++e;
      return e;
    }
    size_t w = i;
    while (w < n && isS(w)) ++w;
    // 5: \s* [\r\n]+  -> through the last CR / LF of the whitespace run
    for (size_t k = w; k > i; --k)
      if (crlf(k - 1)) return k;
    if (w > i) {
      if (w == n) return w;            // 6: \s+ (?!\S) at the end of the text
      if (w - i >= 2) return w - 1;    // 6: all but the last whitespace
      return w;                        // 7: \s+
    }
    return i + 1;
  };

  pieces.clear();
  for (size_t i = 0; i < n;) {
    size_t e = match(i);
    pieces.push_back({off[i], off[e]});
    i = e;
  }
}

// GPT-2's bytes_to_unicode: printable bytes stand for themselves, the others for 256 + k
void byte_chars(std::string (&out)[256]) {
  std::vector<int> bs, cs;
  for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
  for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
  for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
  cs = bs;
  int k = 0;
  for (int b = 0; b < 256; ++b)
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + k++);
    }
  for (size_t i = 0; i < bs.size(); ++i) {
    uint32_t c = uint32_t(cs[i]);
    std::string u;
    if (c < 0x80)
      u += char(c);
    else {
      u += char(0xC0 | (c >> 6));
      u += char(0x80 | (c & 0x3F));
    }
    out[bs[i]] = u;
  }
}

uint64_t pair_key(int32_t a, int32_t b) { return (uint64_t(uint32_t(a)) << 32) | uint32_t(b); }

}  // namespace

Tokenizer::Tokenizer(const std::string& path) {
  std::ifstream f(path);
  XTG_CHECK(f.good(), "cannot read %s", path.c_str());
  nlohmann::json t = nlohmann::json::parse(f);
  const auto& model = t.at("model");
  XTG_CHECK(model.value("type", "") == "BPE", "%s: only BPE tokenizers are supported", path.c_str());
  auto unset = [](const nlohmann::json& j, const char* k) { return !j.contains(k) || j[k].is_null(); };
  XTG_CHECK(!model.value("byte_fallback", false) && unset(model, "continuing_subword_prefix") &&
                unset(model, "end_of_word_suffix"),
            "%s: unsupported BPE options", path.c_str());
  ignore_merges_ = model.value("ignore_merges", false);
  XTG_CHECK(unset(t, "normalizer"), "%s: normalizers are not supported", path.c_str());
  const auto& pre = t.at("pre_tokenizer");
  bool ok = pre.value("type", "") == "Sequence" && pre.contains("pretokenizers") && pre.at("pretokenizers").size() == 2;
  if (ok) {
    const auto& split = pre.at("pretokenizers").at(0);
    const auto& bl = pre.at("pretokenizers").at(1);
    ok = split.value("type", "") == "Split" && split.contains("pattern") &&
         split.at("pattern").value("Regex", "") == kPattern &&
         split.value("behavior", "") == "Isolated" && !split.value("invert", true) && bl.value("type", "") == "ByteLevel" &&
         !bl.value("add_prefix_space", true) && !bl.value("use_regex", true);
  }
  XTG_CHECK(ok, "%s: the pre-tokenizer is not the byte-level Tekken one this tokenizer implements", path.c_str());

  byte_chars(byte_char_);
  std::unordered_map<std::string, unsigned char> char_byte;
  for (int b = 0; b < 256; ++b) char_byte[byte_char_[b]] = static_cast<unsigned char>(b);

  int32_t max_id = -1;
  for (auto it = model["vocab"].begin(); it != model["vocab"].end(); ++it) max_id = std::max(max_id, it.value().get<int32_t>());
  for (const auto& a : t["added_tokens"]) max_id = std::max(max_id, a["id"].get<int32_t>());
  bytes_.assign(size_t(max_id + 1), std::string());
  special_.assign(size_t(max_id + 1), false);
  added_.assign(size_t(max_id + 1), false);
  for (auto it = model["vocab"].begin(); it != model["vocab"].end(); ++it) {
    const std::string& s = it.key();
    int32_t id = it.value().get<int32_t>();
    vocab_[s] = id;
    // byte-level characters are one or two UTF-8 bytes each
    std::string raw;
    for (size_t i = 0; i < s.size();) {
      size_t len = (static_cast<unsigned char>(s[i]) < 0x80) ? 1 : 2;
      auto c = char_byte.find(s.substr(i, len));
      XTG_CHECK(c != char_byte.end(), "%s: vocabulary entry %d is not byte-level", path.c_str(), id);
      raw += char(c->second);
      i += len;
    }
    bytes_[size_t(id)] = raw;
  }
  int32_t rank = 0;
  for (const auto& m : model["merges"]) {
    std::string a, b;
    if (m.is_array()) {
      a = m[0].get<std::string>();
      b = m[1].get<std::string>();
    } else {
      std::string s = m.get<std::string>();
      size_t sp = s.find(' ');
      a = s.substr(0, sp);
      b = s.substr(sp + 1);
    }
    auto ia = vocab_.find(a), ib = vocab_.find(b), im = vocab_.find(a + b);
    XTG_CHECK(ia != vocab_.end() && ib != vocab_.end() && im != vocab_.end(), "%s: merge %d has unknown tokens",
              path.c_str(), rank);
    merges_.emplace(pair_key(ia->second, ib->second), std::make_pair(rank, im->second));
    ++rank;
  }
  for (const auto& a : t["added_tokens"]) {
    int32_t id = a["id"].get<int32_t>();
    std::string content = a["content"].get<std::string>();
    XTG_CHECK(!a.value("lstrip", false) && !a.value("rstrip", false) && !a.value("single_word", false) &&
                  !a.value("normalized", true),
              "%s: added token %d has options this tokenizer does not implement", path.c_str(), id);
    bytes_[size_t(id)] = content;
    added_[size_t(id)] = true;
    special_[size_t(id)] = a.value("special", false);
    added_list_.push_back({content, id});
  }
  std::stable_sort(added_list_.begin(), added_list_.end(),
                   [](const auto& x, const auto& y) { return x.first.size() > y.first.size(); });
  for (size_t i = 0; i < added_list_.size(); ++i)
    if (!added_list_[i].first.empty()) first_[static_cast<unsigned char>(added_list_[i].first[0])].push_back(i);
}

const std::string& Tokenizer::token_bytes(int32_t id) const {
  static const std::string empty;
  return id >= 0 && id < int32_t(bytes_.size()) ? bytes_[size_t(id)] : empty;
}

int32_t Tokenizer::find(const std::string& content) const {
  for (const auto& a : added_list_)
    if (a.first == content) return a.second;
  std::string bl;
  for (unsigned char c : content) bl += byte_char_[c];
  auto it = vocab_.find(bl);
  return it == vocab_.end() ? -1 : it->second;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids, bool skip_special) const {
  std::string s;
  for (int32_t id : ids) {
    if (skip_special && is_special(id)) continue;
    s += token_bytes(id);
  }
  return s;
}

void Tokenizer::encode_piece(const std::string& piece, std::vector<int32_t>& out) const {
  std::string bl;
  for (unsigned char c : piece) bl += byte_char_[c];
  if (ignore_merges_) {
    auto it = vocab_.find(bl);
    if (it != vocab_.end()) {
      out.push_back(it->second);
      return;
    }
  }
  struct Sym {
    int32_t id;
    int prev, next, len;
  };
  std::vector<Sym> syms;
  syms.reserve(piece.size());
  for (size_t i = 0; i < piece.size(); ++i) {
    auto it = vocab_.find(byte_char_[static_cast<unsigned char>(piece[i])]);
    XTG_CHECK(it != vocab_.end(), "byte %u has no token", unsigned(static_cast<unsigned char>(piece[i])));
    syms.push_back({it->second, int(i) - 1, i + 1 < piece.size() ? int(i) + 1 : -1, 1});
  }
  struct Merge {
    int pos;
    int32_t rank, id;
  };
  auto later = [](const Merge& a, const Merge& b) { return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos; };
  std::priority_queue<Merge, std::vector<Merge>, decltype(later)> queue(later);
  auto push = [&](int pos) {
    if (pos < 0 || syms[size_t(pos)].next < 0) return;
    auto it = merges_.find(pair_key(syms[size_t(pos)].id, syms[size_t(syms[size_t(pos)].next)].id));
    if (it != merges_.end()) queue.push({pos, it->second.first, it->second.second});
  };
  for (int i = 0; i + 1 < int(syms.size()); ++i) push(i);
  // the merge loop of Hugging Face tokenizers (word.rs merge_all): lowest rank first, then leftmost
  while (!queue.empty()) {
    Merge top = queue.top();
    queue.pop();
    Sym& cur = syms[size_t(top.pos)];
    if (cur.len == 0 || cur.next < 0) continue;
    Sym& right = syms[size_t(cur.next)];
    auto it = merges_.find(pair_key(cur.id, right.id));
    if (it == merges_.end() || it->second.second != top.id) continue;
    cur.id = top.id;
    cur.len += right.len;
    cur.next = right.next;
    right.len = 0;
    if (cur.next >= 0) syms[size_t(cur.next)].prev = top.pos;
    push(cur.prev);
    push(top.pos);
  }
  for (const Sym& s : syms)
    if (s.len > 0) out.push_back(s.id);
}

void Tokenizer::encode_plain(const std::string& text, std::vector<int32_t>& out) const {
  if (text.empty()) return;
  std::vector<std::pair<size_t, size_t>> pieces;
  split_pieces(text, pieces);
  for (auto [b, e] : pieces) encode_piece(text.substr(b, e - b), out);
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool parse_special) const {
  std::vector<int32_t> out;
  if (!parse_special) {
    encode_plain(text, out);
    return out;
  }
  size_t start = 0, i = 0;
  while (i < text.size()) {
    const std::pair<std::string, int32_t>* hit = nullptr;
    for (size_t k : first_[static_cast<unsigned char>(text[i])]) {   // longest first: the first that fits
      const auto& a = added_list_[k];
      if (text.compare(i, a.first.size(), a.first) == 0) {
        hit = &a;
        break;
      }
    }
    if (!hit) {
      ++i;
      continue;
    }
    encode_plain(text.substr(start, i - start), out);
    out.push_back(hit->second);
    i += hit->first.size();
    start = i;
  }
  encode_plain(text.substr(start), out);
  return out;
}

}  // namespace xtg
