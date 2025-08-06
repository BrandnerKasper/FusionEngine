#pragma once
#include <filesystem>

#define GET_STRING(x) #x
#define GET_DIR(x) GET_STRING(x)

namespace fs = std::filesystem;

namespace assets {
    // TODO: use local asset folder -> and implement shader hot reloading.
    inline fs::path path(const fs::path &relativeAssetPath) {
        const auto mergedPath = (GET_DIR(ASSET_ROOT) / relativeAssetPath).make_preferred();
        return fs::canonical(mergedPath);
    }
}