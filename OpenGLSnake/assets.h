#pragma once
#include <filesystem>


namespace fs = std::filesystem;

namespace assets {
    inline fs::path path(const fs::path &relativeAssetPath) {
        const auto mergedPath = (ASSET_ROOT / relativeAssetPath).make_preferred();
        return fs::canonical(mergedPath);
    }
}