#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cstdint>
#include <regex>
#include <cmath>
#include <algorithm>

namespace fs = std::filesystem;

#pragma pack(push, 1)
// BMP file headers
struct BMPFileHeader {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};

struct BMPInfoHeader {
    uint32_t biSize;
    int32_t biWidth;
    int32_t biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t biXPelsPerMeter;
    int32_t biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};
#pragma pack(pop)

// Pixel structures
struct Pixel24 {
    uint8_t blue, green, red;
};

// Read BMP file
template <typename PixelType>
bool Reading(const std::string& filename, BMPFileHeader& fileHeader, BMPInfoHeader& infoHeader, std::vector<PixelType>& pixels) {
    std::ifstream file(filename, std::ios::binary);

    file.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
    file.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));

    std::cout << filename << " \nBit Depth: " << infoHeader.biBitCount << std::endl;

    int width = infoHeader.biWidth;
    int height = infoHeader.biHeight;
    pixels.resize(width * height);

    file.seekg(fileHeader.bfOffBits, std::ios::beg);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            file.read(reinterpret_cast<char*>(&pixels[(height - 1 - y) * width + x]), sizeof(PixelType));
        }
    }

    return true;
}

// Write BMP file
template <typename PixelType>
bool Writing(const std::string& filename, const BMPFileHeader& fileHeader, const BMPInfoHeader& infoHeader, const std::vector<PixelType>& pixels) {
    std::ofstream file(filename, std::ios::binary);

    file.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));

    int width = infoHeader.biWidth;
    int height = infoHeader.biHeight;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            file.write(reinterpret_cast<const char*>(&pixels[(height - 1 - y) * width + x]), sizeof(PixelType));
        }
    }

    return true;
}

// Function to adjust color temperature
void ColorTemperature(std::vector<Pixel24>& pixels, double redFactor, double greenFactor, double blueFactor) {
    for (auto& pixel : pixels) {
        pixel.red = std::clamp(static_cast<int>(pixel.red * redFactor), 0, 255);
        pixel.green = std::clamp(static_cast<int>(pixel.green * greenFactor), 0, 255);
        pixel.blue = std::clamp(static_cast<int>(pixel.blue * blueFactor), 0, 255);
    }
}

// Generate Output File Name
std::string OutputName(const std::string& inputFile, int temperatureSuffix) {
    std::regex re("(output\\d+)_\\d+\\.bmp"); // Matches filenames like outputX_Y.bmp
    std::smatch match;

    if (std::regex_match(inputFile, match, re) && match.size() == 2) {
        std::string baseName = match[1]; // Extract "outputX"
        return baseName + "_" + std::to_string(temperatureSuffix) + ".bmp";
    }

    // If the input filename doesn't match the expected pattern, return a default
    return "output_" + std::to_string(temperatureSuffix) + ".bmp";
}

// Process Batch of Images
void Processing(const std::vector<std::string>& inputFiles) {
    for (size_t i = 0; i < inputFiles.size(); ++i) {
        BMPFileHeader fileHeader;
        BMPInfoHeader infoHeader;

        // Read BMP file headers
        std::vector<Pixel24> pixels;
        if (!Reading(inputFiles[i], fileHeader, infoHeader, pixels)) {
            std::cerr << "Error reading file: " << inputFiles[i] << std::endl;
            continue;
        }

        // Ensure we have 24-bit images
        if (infoHeader.biBitCount != 24) {
            std::cerr << "Invalid bit depth in file: " << inputFiles[i] << std::endl;
            continue;
        }

        // Warm adjustment
        std::vector<Pixel24> warmPixels = pixels; // Copy of original pixels
        ColorTemperature(warmPixels, 1.2, 1.0, 0.8); // Slightly warm tones
        std::string warmOutput = OutputName(inputFiles[i], 3); // Suffix 3 for warm
        Writing(warmOutput, fileHeader, infoHeader, warmPixels);

        // Cool adjustment
        std::vector<Pixel24> coolPixels = pixels; // Copy of original pixels
        ColorTemperature(coolPixels, 0.8, 1.0, 1.2); // Slightly cool tones
        std::string coolOutput = OutputName(inputFiles[i], 4); // Suffix 4 for cool
        Writing(coolOutput, fileHeader, infoHeader, coolPixels);
    }
}

int main() {
    // Input files
    std::vector<std::string> inputFiles = {"output1_2.bmp", "output2_2.bmp", "output3_2.bmp", "output4_2.bmp"};

    // Process the files (warm and cool versions will be generated)
    Processing(inputFiles);

    return 0;
}
