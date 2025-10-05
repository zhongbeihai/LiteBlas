#ifndef CMDLINE_H
#define CMDLINE_H

#include <unordered_set>
#include <unordered_map>
#include <string>
#include <vector>

class CommandLineOptions {
public:

    CommandLineOptions(int argc, char* argv[]) { parse(argc, argv); }
    bool illegal_present() const;

    bool help()    const { return has_flag("help"); }
    bool verbose() const { return has_flag("verb"); }
    bool noverif() const { return has_flag("noverif"); }
    bool get_debug()   const { return has_flag("debug"); }

    int size_n(int def = 256) const { return get_int("n", def); }
    int reps(int def = 100) const { return get_int("reps", def); }
    int get_seed(int def = 1) const { return get_int("seed", def); }
    std::string kernel(const std::string& def = "naive_ijk") const {
        return get_option("kernel", def);
    }

private:

    std::unordered_map<std::string, std::string> options;
    std::unordered_set<std::string> flags;
    std::vector<std::string> positional;

    std::string normalize_opt(const std::string& k);
    std::string normalize_flag(const std::string& k);
    int get_int(const std::string& key, int def) const;

    void parse(int argc, char* argv[]);

    bool has_flag(const std::string key) const { return flags.count(key) > 0; }
    bool has_option(const std::string key) const { return options.count(key) > 0; }

    std::string get_option(const std::string& key, const std::string& def = "") const {
        auto it = options.find(key);
        return it != options.end() ? it->second : def;
    }
};

#endif // CMDLINE_H
