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

void GammaCorrection(std::vector<Pixel24>& pixels, double gammaRed, double gammaGreen, double gammaBlue) {
    for (auto& pixel : pixels) {
        pixel.red = std::min(255, static_cast<int>(pow(pixel.red / 255.0, 1.0 / gammaRed) * 255.0));
        pixel.green = std::min(255, static_cast<int>(pow(pixel.green / 255.0, 1.0 / gammaGreen) * 255.0));
        pixel.blue = std::min(255, static_cast<int>(pow(pixel.blue / 255.0, 1.0 / gammaBlue) * 255.0));
    }
}

void Saturation(std::vector<Pixel24>& pixels, double factor) {
    for (auto& pixel : pixels) {
        // Convert RGB to HSV
        double r = pixel.red / 255.0;
        double g = pixel.green / 255.0;
        double b = pixel.blue / 255.0;

        double maxVal = std::max({r, g, b});
        double minVal = std::min({r, g, b});
        double delta = maxVal - minVal;

        double saturation = (maxVal == 0) ? 0 : delta / maxVal;
        double value = maxVal;

        // Adjust saturation
        saturation = std::clamp(saturation * factor, 0.0, 1.0);

        // Convert back to RGB
        if (delta == 0) {
            r = g = b = value;
        } else {
            double h;
            if (maxVal == r) h = (g - b) / delta;
            else if (maxVal == g) h = 2.0 + (b - r) / delta;
            else h = 4.0 + (r - g) / delta;

            h = std::fmod(h + 6.0, 6.0);
            double c = value * saturation;
            double x = c * (1 - std::abs(std::fmod(h, 2.0) - 1));
            double m = value - c;

            if (0 <= h && h < 1) { r = c; g = x; b = 0; }
            else if (1 <= h && h < 2) { r = x; g = c; b = 0; }
            else if (2 <= h && h < 3) { r = 0; g = c; b = x; }
            else if (3 <= h && h < 4) { r = 0; g = x; b = c; }
            else if (4 <= h && h < 5) { r = x; g = 0; b = c; }
            else { r = c; g = 0; b = x; }

            r += m;
            g += m;
            b += m;
        }

        pixel.red = std::min(255, static_cast<int>(r * 255));
        pixel.green = std::min(255, static_cast<int>(g * 255));
        pixel.blue = std::min(255, static_cast<int>(b * 255));
    }
}

void Contrast(std::vector<Pixel24>& pixels, double factorRed, double factorGreen, double factorBlue) {
    double avgIntensityRed = 0, avgIntensityGreen = 0, avgIntensityBlue = 0;
    for (const auto& pixel : pixels) {
        avgIntensityRed += pixel.red;
        avgIntensityGreen += pixel.green;
        avgIntensityBlue += pixel.blue;
    }

    avgIntensityRed /= pixels.size();
    avgIntensityGreen /= pixels.size();
    avgIntensityBlue /= pixels.size();

    for (auto& pixel : pixels) {
        pixel.red = std::min(255, std::max(0, static_cast<int>(avgIntensityRed + factorRed * (pixel.red - avgIntensityRed))));
        pixel.green = std::min(255, std::max(0, static_cast<int>(avgIntensityGreen + factorGreen * (pixel.green - avgIntensityGreen))));
        pixel.blue = std::min(255, std::max(0, static_cast<int>(avgIntensityBlue + factorBlue * (pixel.blue - avgIntensityBlue))));
    }
}

// Generate Output File Name
std::string OutputName(const std::string& inputFile) {
    std::regex re("output(\\d+)(_(\\d+))?\\.bmp");  // Match outputX_Y.bmp
    std::smatch match;
    if (std::regex_search(inputFile, match, re) && match.size() > 1) {
        std::string outputBase = "output" + match.str(1);  // outputX
        std::string suffix = match.str(2);  // _Y (if present)
        int newSuffix = 2;  // Add the new suffix to the name (hardcoded to 2 for simplicity)

        return outputBase + "_" + std::to_string(newSuffix) + ".bmp";
    }
    return "output.bmp";
}

void Processing(const std::vector<std::string>& inputFiles, const std::vector<std::string>& outputFiles) {
    for (size_t i = 0; i < inputFiles.size(); ++i) {
        BMPFileHeader fileHeader;
        BMPInfoHeader infoHeader;

        std::ifstream file(inputFiles[i], std::ios::binary);
        file.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
        file.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));
        file.close();

        if (infoHeader.biBitCount == 24) {
            std::vector<Pixel24> pixels;
            if (!Reading(inputFiles[i], fileHeader, infoHeader, pixels)) continue;

            // Apply adjustments specific to each image
            if (inputFiles[i] == "output1_1.bmp") {
                // Adjust image properties
                GammaCorrection(pixels, 1.6, 1.8, 2.0);  // Tweak per channel
                Contrast(pixels, 1.4, 1.5, 1.6);        // Tweak contrast
                Saturation(pixels, 1.5);                // Adjust saturation
            }
            else if (inputFiles[i] == "output2_1.bmp") {
                // Adjust image properties
                GammaCorrection(pixels, 1.8, 2.0, 2.5);  // Tweak per channel
                Contrast(pixels, 1.0, 1.3, 1.5);        // Tweak contrast
                Saturation(pixels, 2.2);                // Adjust saturation
            } 
            else if (inputFiles[i] == "output3_1.bmp") {
                // Adjust image properties
                GammaCorrection(pixels, 1.6, 1.6, 1.6);  // Tweak per channel
                Contrast(pixels, 1.3, 1.1, 1.2);        // Tweak contrast
                Saturation(pixels, 1.5);                // Adjust saturation
            }
            else if (inputFiles[i] == "output4_1.bmp") {
               // Adjust image properties
                GammaCorrection(pixels, 0.6, 0.5, 0.4);  // Tweak per channel
                Contrast(pixels, 0.9, 0.7, 0.8);        // Tweak contrast
                Saturation(pixels, 1.2);                // Adjust saturation
            }

            Writing(outputFiles[i], fileHeader, infoHeader, pixels);
        }
    }
}

int main() {
    std::vector<std::string> inputFiles = {"output1_1.bmp", "output2_1.bmp", "output3_1.bmp", "output4_1.bmp"};
    std::vector<std::string> outputFiles;

    for (const auto& inputFile : inputFiles) {
        outputFiles.push_back(OutputName(inputFile));
    }
    Processing(inputFiles, outputFiles);

    return 0;
}
