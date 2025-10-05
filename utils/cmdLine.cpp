#include "cmdLine.h"

std::string CommandLineOptions::normalize_opt(const std::string& k) {
    if (k == "n" || k == "size")   return "n";
    if (k == "r" || k == "reps")   return "reps";
    if (k == "kernel" || k == "k") return "kernel";
    if (k == "seed" || k == "s")   return "seed";
    return k;
}

std::string CommandLineOptions::normalize_flag(const std::string& k) {
    if (k == "h" || k == "help")  return "help";
    if (k == "v" || k == "verb")  return "verb";
    if (k == "noverif")           return "noverif";
    if (k == "debug" || k == "d") return "debug";
    return k;
}

int CommandLineOptions::get_int(const std::string& key, int def) const {
    auto it = options.find(key);
    if (it == options.end()) return def;
    try {
        return std::stoi(it->second);
    } catch (...) {
        return def;
    }
    return def;
}

void CommandLineOptions::parse(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) == 0) {
            auto eq = arg.find('=');
            if (eq != std::string::npos) {
                std::string key = normalize_opt(arg.substr(2, eq - 2));
                std::string val = arg.substr(eq + 1);
                options[key] = val;
            } else {
                std::string key = arg.substr(2);
                std::string canon_opt = normalize_opt(key);
                std::string canon_flag = normalize_flag(key);
                if (canon_opt == "n" || canon_opt == "reps" || canon_opt == "kernel" || canon_opt == "seed") {
                    if (i + 1 < argc && argv[i + 1][0] != '-') {
                        options[canon_opt] = argv[++i];
                    } else {
                        flags.insert(canon_opt);
                    }
                } else {
                    flags.insert(canon_flag);
                }
            }
        } else if (arg.rfind("-", 0) == 0 && arg.size() > 1) {
            auto eq = arg.find('=');
            if (eq != std::string::npos) {
                std::string key = normalize_opt(std::string(1, arg[eq - 1]));
                std::string val = arg.substr(eq + 1);
                options[key] = val;
            } else {
                std::string raw(1, arg[1]);
                std::string kopt  = normalize_opt(raw);
                std::string kflag = normalize_flag(raw);
                if (kopt == "n" || kopt == "reps" || kopt == "kernel" || kopt == "seed") {
                    if (i + 1 < argc && argv[i + 1][0] != '-') {
                        options[kopt] = argv[++i];
                    } else {
                        flags.insert(kopt);
                    }
                } else {
                    flags.insert(kflag);
                }
            }
        } else {
            positional.push_back(arg);
        }
    }
}

bool CommandLineOptions::illegal_present() const {
    static const std::unordered_set<std::string> legal_flags  = {
        "help", "verb", "noverif", "debug"
    };
    static const std::unordered_set<std::string> legal_opts   = {
        "n", "reps", "kernel", "seed"
    };
    for (const auto& f : flags) {
        if (!legal_flags.count(f)) return true;
    }
    for (const auto& kv : options) {
        if (!legal_opts.count(kv.first)) return true;
    }
    if (auto it = options.find("kernel"); it != options.end()) {
        static const std::unordered_set<std::string> allowed = {
            "naive_ijk","openblas","blislab","my_kernel"
        };
        if (!allowed.count(it->second)) return true;
    }
    return false;
}
