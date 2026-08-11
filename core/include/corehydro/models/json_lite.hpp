// corehydro ADDITION -- no upstream C# counterpart.
//
// A minimal, self-contained JSON reader used by the shared fixture model-spec builder
// (models/model_spec.hpp). The three fixture harnesses each hold a `construct.model`
// object in their native form (nlohmann::json in the C++ test runner, a jsonlite list in
// R, a dict in Python); rather than re-implementing the model-construction logic three
// times -- or teaching the R/Python packages a third-party JSON dependency (the core must
// stay dependency-free for CRAN/PyPI) -- each harness re-serializes the spec to a JSON
// string and hands it to ONE shared C++ entry point, which parses it with this reader.
//
// Deliberately small: parse-only (no writer), values are immutable after parsing, numbers
// are always double (fixture specs carry no 64-bit integers), objects preserve insertion
// order in a flat key/value vector (specs are tiny -- no hashing needed). Standard JSON
// escapes including \uXXXX (BMP code points; surrogate pairs are rejected -- fixture specs
// are ASCII). Errors throw std::runtime_error with a byte offset.
//
// NOT a general-purpose JSON library: no comments, no trailing commas, no NaN/Infinity
// literals (fixture model specs encode plain finite numbers only; the runners' "inf"/"nan"
// STRING convention applies to assertion values, which never pass through this reader).
#pragma once
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace corehydro::models::spec {

class JsonValue {
   public:
    enum class Type { Null, Boolean, Number, String, Array, Object };

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }

    bool as_bool() const {
        require(Type::Boolean, "boolean");
        return boolean_;
    }
    double as_double() const {
        require(Type::Number, "number");
        return number_;
    }
    int as_int() const { return static_cast<int>(as_double()); }
    const std::string& as_string() const {
        require(Type::String, "string");
        return string_;
    }
    const std::vector<JsonValue>& items() const {
        require(Type::Array, "array");
        return array_;
    }

    bool contains(const std::string& key) const {
        if (type_ != Type::Object) return false;
        for (const auto& kv : object_)
            if (kv.first == key) return true;
        return false;
    }
    const JsonValue& at(const std::string& key) const {
        require(Type::Object, "object");
        for (const auto& kv : object_)
            if (kv.first == key) return kv.second;
        throw std::runtime_error("model spec: missing required key '" + key + "'");
    }

    // Typed defaults for optional keys.
    std::string value_or(const std::string& key, const char* dflt) const {
        return contains(key) ? at(key).as_string() : std::string(dflt);
    }
    double value_or(const std::string& key, double dflt) const {
        return contains(key) ? at(key).as_double() : dflt;
    }
    int value_or(const std::string& key, int dflt) const {
        return contains(key) ? at(key).as_int() : dflt;
    }
    bool value_or(const std::string& key, bool dflt) const {
        return contains(key) ? at(key).as_bool() : dflt;
    }

    // Convenience: a JSON array of numbers -> std::vector<double>.
    std::vector<double> as_double_vector() const {
        std::vector<double> out;
        out.reserve(items().size());
        for (const JsonValue& v : items()) out.push_back(v.as_double());
        return out;
    }

    // The object's key/value pairs in insertion (= source) order. Needed by to_json_string
    // (below) and by fit_runner.hpp's fitted_spec, which re-emits an object's own entries plus
    // one extra key.
    const std::vector<std::pair<std::string, JsonValue>>& entries() const {
        require(Type::Object, "object");
        return object_;
    }

   private:
    friend class JsonParser;

    void require(Type expected, const char* name) const {
        if (type_ != expected)
            throw std::runtime_error(std::string("model spec: expected a JSON ") + name);
    }

    Type type_ = Type::Null;
    bool boolean_ = false;
    double number_ = 0.0;
    std::string string_;
    std::vector<JsonValue> array_;
    std::vector<std::pair<std::string, JsonValue>> object_;
};

class JsonParser {
   public:
    static JsonValue parse(const std::string& text) {
        JsonParser p(text);
        p.skip_ws();
        JsonValue v = p.parse_value();
        p.skip_ws();
        if (p.pos_ != p.s_.size()) p.fail("trailing characters after JSON value");
        return v;
    }

   private:
    explicit JsonParser(const std::string& text) : s_(text) {}

    [[noreturn]] void fail(const std::string& what) const {
        throw std::runtime_error("model spec JSON error at offset " + std::to_string(pos_) +
                                 ": " + what);
    }

    void skip_ws() {
        while (pos_ < s_.size() &&
               (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r'))
            ++pos_;
    }

    char peek() const {
        if (pos_ >= s_.size())
            throw std::runtime_error("model spec JSON error: unexpected end of input");
        return s_[pos_];
    }

    void expect(char c) {
        if (peek() != c) fail(std::string("expected '") + c + "'");
        ++pos_;
    }

    bool consume_literal(const char* lit) {
        std::size_t n = 0;
        while (lit[n] != '\0') ++n;
        if (s_.compare(pos_, n, lit) != 0) return false;
        pos_ += n;
        return true;
    }

    JsonValue parse_value() {
        switch (peek()) {
            case '{':
                return parse_object();
            case '[':
                return parse_array();
            case '"':
                return parse_string_value();
            case 't':
            case 'f':
                return parse_boolean();
            case 'n':
                if (!consume_literal("null")) fail("invalid literal");
                return JsonValue{};  // Null
            default:
                return parse_number();
        }
    }

    JsonValue parse_object() {
        expect('{');
        JsonValue v;
        v.type_ = JsonValue::Type::Object;
        skip_ws();
        if (peek() == '}') {
            ++pos_;
            return v;
        }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(':');
            skip_ws();
            v.object_.emplace_back(std::move(key), parse_value());
            skip_ws();
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            expect('}');
            return v;
        }
    }

    JsonValue parse_array() {
        expect('[');
        JsonValue v;
        v.type_ = JsonValue::Type::Array;
        skip_ws();
        if (peek() == ']') {
            ++pos_;
            return v;
        }
        while (true) {
            skip_ws();
            v.array_.push_back(parse_value());
            skip_ws();
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            expect(']');
            return v;
        }
    }

    JsonValue parse_boolean() {
        JsonValue v;
        v.type_ = JsonValue::Type::Boolean;
        if (consume_literal("true")) {
            v.boolean_ = true;
        } else if (consume_literal("false")) {
            v.boolean_ = false;
        } else {
            fail("invalid literal");
        }
        return v;
    }

    JsonValue parse_number() {
        std::size_t start = pos_;
        while (pos_ < s_.size() &&
               (s_[pos_] == '-' || s_[pos_] == '+' || s_[pos_] == '.' || s_[pos_] == 'e' ||
                s_[pos_] == 'E' || (s_[pos_] >= '0' && s_[pos_] <= '9')))
            ++pos_;
        if (pos_ == start) fail("invalid JSON value");
        const std::string token = s_.substr(start, pos_ - start);
        char* end = nullptr;
        // NOTE (M13 ledger): std::strtod is LC_NUMERIC locale-sensitive -- a host that
        // setlocale()s a comma-decimal numeric locale would misparse "1.5" (nlohmann/json
        // works around this; json_lite does not). Acceptable for the fixture-harness path
        // only: the R runner pins LC_NUMERIC=C and the Python/C++ test hosts never call
        // setlocale(). Revisit with a locale-independent parse (e.g. std::from_chars)
        // if json_lite ever leaves the test/spec-builder surface.
        double value = std::strtod(token.c_str(), &end);
        if (end != token.c_str() + token.size()) fail("invalid number '" + token + "'");
        JsonValue v;
        v.type_ = JsonValue::Type::Number;
        v.number_ = value;
        return v;
    }

    JsonValue parse_string_value() {
        JsonValue v;
        v.type_ = JsonValue::Type::String;
        v.string_ = parse_string();
        return v;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            if (pos_ >= s_.size()) fail("unterminated string");
            char c = s_[pos_++];
            if (c == '"') return out;
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (pos_ >= s_.size()) fail("unterminated escape");
            char e = s_[pos_++];
            switch (e) {
                case '"':
                case '\\':
                case '/':
                    out.push_back(e);
                    break;
                case 'b':
                    out.push_back('\b');
                    break;
                case 'f':
                    out.push_back('\f');
                    break;
                case 'n':
                    out.push_back('\n');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case 'u':
                    append_unicode_escape(out);
                    break;
                default:
                    fail("invalid escape");
            }
        }
    }

    void append_unicode_escape(std::string& out) {
        if (pos_ + 4 > s_.size()) fail("truncated \\u escape");
        unsigned int cp = 0;
        for (int i = 0; i < 4; ++i) {
            char c = s_[pos_++];
            cp <<= 4;
            if (c >= '0' && c <= '9')
                cp |= static_cast<unsigned int>(c - '0');
            else if (c >= 'a' && c <= 'f')
                cp |= static_cast<unsigned int>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F')
                cp |= static_cast<unsigned int>(c - 'A' + 10);
            else
                fail("invalid \\u escape");
        }
        if (cp >= 0xD800 && cp <= 0xDFFF)
            fail("surrogate pairs are not supported (fixture specs are ASCII)");
        // UTF-8 encode the BMP code point.
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    const std::string& s_;
    std::size_t pos_ = 0;
};

inline JsonValue parse_json(const std::string& text) { return JsonParser::parse(text); }

// --- Serialization (corehydro ADDITION; json_lite.hpp is otherwise parse-only) --------------
//
// A small writer, added for fit_runner.hpp's `fitted_spec`: the runner needs to hand a caller
// back the parsed `construct.model` object with the fitted values spliced in, and the only way
// to re-emit an already-parsed JsonValue is to serialize it ourselves (no third-party writer is
// available -- see the file header on staying dependency-free). Round-trips numbers via
// `%.17g`, matching the `spec_number` contract the R glue's `jsonlite::toJSON(digits = I(17))`
// already relies on for exact double round-tripping.
inline std::string escape_json_string(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

inline std::string to_json_string(const JsonValue& v) {
    switch (v.type()) {
        case JsonValue::Type::Null:
            return "null";
        case JsonValue::Type::Boolean:
            return v.as_bool() ? "true" : "false";
        case JsonValue::Type::Number: {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.17g", v.as_double());
            return std::string(buf);
        }
        case JsonValue::Type::String:
            return "\"" + escape_json_string(v.as_string()) + "\"";
        case JsonValue::Type::Array: {
            std::string out = "[";
            const auto& items = v.items();
            for (std::size_t i = 0; i < items.size(); ++i) {
                if (i != 0) out += ",";
                out += to_json_string(items[i]);
            }
            out += "]";
            return out;
        }
        case JsonValue::Type::Object: {
            std::string out = "{";
            const auto& entries = v.entries();
            for (std::size_t i = 0; i < entries.size(); ++i) {
                if (i != 0) out += ",";
                out += "\"" + escape_json_string(entries[i].first) + "\":" +
                       to_json_string(entries[i].second);
            }
            out += "}";
            return out;
        }
    }
    return "null";  // unreachable; silences -Wreturn-type on some compilers
}

}  // namespace corehydro::models::spec
