#pragma once
#include <filesystem>


namespace fs = std::filesystem;

namespace assets {
    inline fs::path path(const fs::path& relativeAssetPath) {
        const auto mergedPath = (ASSET_ROOT / relativeAssetPath).make_preferred();
        return fs::canonical(mergedPath);
    }
}

namespace data {
    inline fs::path path(const fs::path& relativeDataPath) {
        const auto mergedPath = (DATA_ROOT / relativeDataPath).make_preferred();
        return fs::weakly_canonical(mergedPath);
    }
}

// TODO: add a namespace model or neural model