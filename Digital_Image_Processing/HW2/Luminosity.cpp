#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <regex>

#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType{0x4D42};  // "BM" in little-endian
    uint32_t fileSize{0};
    uint16_t reserved1{0};
    uint16_t reserved2{0};
    uint32_t offsetData{0};
};

struct DIBHeader {
    uint32_t size{0};
    int32_t width{0};
    int32_t height{0};
    uint16_t planes{1};
    uint16_t bitCount{0}; // Indicates 24-bit or 32-bit
    uint32_t compression{0};
    uint32_t imageSize{0};
    int32_t xPixelsPerMeter{0};
    int32_t yPixelsPerMeter{0};
    uint32_t colorsUsed{0};
    uint32_t colorsImportant{0};
};
#pragma pack(pop)

// Adjust luminosity of a pixel based on its original values and a specified luminosity enhancement factor
void Luminosity(uint8_t &r, uint8_t &g, uint8_t &b, int luminosityBoost) {
    // Calculate the perceived luminosity of the pixel
    double luminosity = 0.21 * r + 0.72 * g + 0.07 * b;

    // Apply enhancement only if luminosity is below a certain threshold, indicating a dark area
    if (luminosity < 100) { // Threshold for low-luminosity areas
        r = std::min(255, static_cast<int>(r + luminosityBoost * (100 - luminosity) / 100));
        g = std::min(255, static_cast<int>(g + luminosityBoost * (100 - luminosity) / 100));
        b = std::min(255, static_cast<int>(b + luminosityBoost * (100 - luminosity) / 100));
    }
}

bool Enhancement(const char *inputFile, const char *outputFile, int luminosityBoost) {
    // Open input BMP file
    std::ifstream inFile(inputFile, std::ios::binary);

    BMPHeader bmpHeader;
    DIBHeader dibHeader;

    // Read BMP header and DIB header
    inFile.read(reinterpret_cast<char*>(&bmpHeader), sizeof(bmpHeader));
    inFile.read(reinterpret_cast<char*>(&dibHeader), sizeof(dibHeader));

    std::cout << inputFile << " \nBit Depth: " << dibHeader.bitCount << std::endl;

    // Calculate row size (considering padding for 24-bit BMP)
    int bytesPerPixel = dibHeader.bitCount / 8;
    int rowSize = ((dibHeader.bitCount * dibHeader.width + 31) / 32) * 4;

    // Allocate memory for pixel data
    std::vector<uint8_t> pixelData(rowSize * dibHeader.height);
    inFile.seekg(bmpHeader.offsetData, std::ios::beg);
    inFile.read(reinterpret_cast<char*>(pixelData.data()), pixelData.size());
    inFile.close();

    // Adjust luminosity in low-luminosity areas
    for (int y = 0; y < dibHeader.height; ++y) {
        for (int x = 0; x < dibHeader.width; ++x) {
            int pixelOffset = y * rowSize + x * bytesPerPixel;
            uint8_t &blue = pixelData[pixelOffset];
            uint8_t &green = pixelData[pixelOffset + 1];
            uint8_t &red = pixelData[pixelOffset + 2];
            Luminosity(red, green, blue, luminosityBoost);

            // For 32-bit BMP, skip the alpha channel
            if (dibHeader.bitCount == 32) {
                // Leave alpha channel unchanged
            }
        }
    }

    // Write output BMP file
    std::ofstream outFile(outputFile, std::ios::binary);

    outFile.write(reinterpret_cast<char*>(&bmpHeader), sizeof(bmpHeader));
    outFile.write(reinterpret_cast<char*>(&dibHeader), sizeof(dibHeader));
    outFile.write(reinterpret_cast<char*>(pixelData.data()), pixelData.size());
    outFile.close();

    return true;
}

std::string OutputName(const std::string &inputFile) {
    // Use a regular expression to find the numeric part after "input" in the filename
    std::regex re("input(\\d+)\\.bmp");
    std::smatch match;
    if (std::regex_search(inputFile, match, re) && match.size() > 1) {
        // Create the output filename with the same number as in the input
        return "output" + match.str(1) + ".bmp";
    }
    // Return a default value if the regex does not match
    return "output.bmp";
}

int main() {
    // List of input BMP files
    std::vector<std::string> inputFiles = {"input1.bmp"};
    int luminosityboost = 50; // Adjust luminosity enhancement level here

    for (const auto &inputFile : inputFiles) {
        // Generate output file name by replacing "input" with "output" and preserving the number
        std::string outputFile = OutputName(inputFile);

        // Call the Enhancement function to process the image
        Enhancement(inputFile.c_str(), outputFile.c_str(), luminosityboost);
    }

    return 0;
}
