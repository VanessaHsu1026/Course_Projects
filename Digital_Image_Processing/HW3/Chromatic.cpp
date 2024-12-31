#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cstdint>
#include <cmath>
#include <regex>

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

// Pixel structure
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

double sRGBtolinear(double value) {
    return (value <= 0.04045) ? (value / 12.92) : std::pow((value + 0.055) / 1.055, 2.4);
}

double lineartosRGB(double value) {
    return (value <= 0.0031308) ? (value * 12.92) : (1.055 * std::pow(value, 1.0 / 2.4) - 0.055);
}

// Apply Gray World Algorithm
void GrayWorld(std::vector<Pixel24>& pixels) {
    // Variables for calculating averages
    double avgR = 0, avgG = 0, avgB = 0;
    size_t totalPixels = pixels.size();

    // Linearize pixel values and calculate averages
    for (const auto& pixel : pixels) {
        double redLinear = sRGBtolinear(pixel.red / 255.0);
        double greenLinear = sRGBtolinear(pixel.green / 255.0);
        double blueLinear = sRGBtolinear(pixel.blue / 255.0);

        avgR += redLinear;
        avgG += greenLinear;
        avgB += blueLinear;
    }

    // Calculate average values
    avgR /= totalPixels;
    avgG /= totalPixels;
    avgB /= totalPixels;

    // Calculate the scaling factors
    double grayValue = (avgR + avgG + avgB) / 3.0;
    double scaleR = grayValue / avgR;
    double scaleG = grayValue / avgG;
    double scaleB = grayValue / avgB;

    // Apply Gray World scaling with gamma correction
    for (auto& pixel : pixels) {
        double redLinear = sRGBtolinear(pixel.red / 255.0) * scaleR;
        double greenLinear = sRGBtolinear(pixel.green / 255.0) * scaleG;
        double blueLinear = sRGBtolinear(pixel.blue / 255.0) * scaleB;

        pixel.red = std::min(255, static_cast<int>(lineartosRGB(redLinear) * 255));
        pixel.green = std::min(255, static_cast<int>(lineartosRGB(greenLinear) * 255));
        pixel.blue = std::min(255, static_cast<int>(lineartosRGB(blueLinear) * 255));
    }
}

// Generate Output File Name
std::string OutputName(const std::string& inputFile) {
    std::regex re("input(\\d+)\\.bmp");
    std::smatch match;
    if (std::regex_search(inputFile, match, re) && match.size() > 1) {
        return "output" + match.str(1) + "_1.bmp";
    }
    return "output.bmp";
}

// Process Batch of Images
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
            GrayWorld(pixels);
            Writing(outputFiles[i], fileHeader, infoHeader, pixels);
        }
    }
}

int main() {
    std::vector<std::string> inputFiles = {"input1.bmp", "input2.bmp", "input3.bmp", "input4.bmp"};
    std::vector<std::string> outputFiles;

    for (const auto& inputFile : inputFiles) {
        outputFiles.push_back(OutputName(inputFile));
    }
    Processing(inputFiles, outputFiles);

    return 0;
}