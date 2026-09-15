#include <iostream> 
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <string>
#include <regex>

#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType{0x4D42};
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
    uint16_t bitCount{0}; // Determines if it's 24-bit or 32-bit
    uint32_t compression{0};
    uint32_t imageSize{0};
    int32_t xPixelsPerMeter{0};
    int32_t yPixelsPerMeter{0};
    uint32_t colorsUsed{0};
    uint32_t colorsImportant{0};
};
#pragma pack(pop)

struct Pixel {
    uint8_t blue;
    uint8_t green;
    uint8_t red;
    uint8_t alpha;
};

// Median Filter function
Pixel MedianFilter(const std::vector<std::vector<Pixel>>& pixels, int x, int y, int kernelSize) {
    int halfKernel = kernelSize / 2;
    std::vector<uint8_t> reds, greens, blues;

    // Gather the neighborhood pixels within the kernel size
    for (int ky = -halfKernel; ky <= halfKernel; ++ky) {
        for (int kx = -halfKernel; kx <= halfKernel; ++kx) {
            int px = std::clamp(x + kx, 0, (int)pixels[0].size() - 1);
            int py = std::clamp(y + ky, 0, (int)pixels.size() - 1);
            reds.push_back(pixels[py][px].red);
            greens.push_back(pixels[py][px].green);
            blues.push_back(pixels[py][px].blue);
        }
    }

    // Sort each color channel and select the median value
    auto median = [](std::vector<uint8_t>& values) {
        std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
        return values[values.size() / 2];
    };

    // Construct the resulting pixel with the median color values
    Pixel result;
    result.red = median(reds);
    result.green = median(greens);
    result.blue = median(blues);
    result.alpha = 255; // Assuming fully opaque

    return result;
}

// Bilateral Filter function
Pixel BilateralFilter(const std::vector<std::vector<Pixel>>& pixels, int x, int y, int kernelSize, float sigmaSpatial, float sigmaColor) {
    int halfKernel = kernelSize / 2;
    float sumR = 0, sumG = 0, sumB = 0;
    float normalizationFactor = 0;

    Pixel centerPixel = pixels[y][x];

    // Process each pixel within the kernel area
    for (int ky = -halfKernel; ky <= halfKernel; ++ky) {
        for (int kx = -halfKernel; kx <= halfKernel; ++kx) {
            int px = std::clamp(x + kx, 0, (int)pixels[0].size() - 1);
            int py = std::clamp(y + ky, 0, (int)pixels.size() - 1);
            Pixel neighborPixel = pixels[py][px];

            // Compute the spatial weight  (based on distance from center)
            float spatialWeight = std::exp(-(kx * kx + ky * ky) / (2 * sigmaSpatial * sigmaSpatial));

            // Compute the color weight based on intensity difference
            float colorDiffR = neighborPixel.red - centerPixel.red;
            float colorDiffG = neighborPixel.green - centerPixel.green;
            float colorDiffB = neighborPixel.blue - centerPixel.blue;
            float colorWeight = std::exp(-(colorDiffR * colorDiffR + colorDiffG * colorDiffG + colorDiffB * colorDiffB) / (2 * sigmaColor * sigmaColor));

            // Combine the weights and update the accumulated color values
            float weight = spatialWeight * colorWeight;
            
            sumR += neighborPixel.red * weight;
            sumG += neighborPixel.green * weight;
            sumB += neighborPixel.blue * weight;
            normalizationFactor += weight;
        }
    }

    // Calculate the resulting pixel values by normalizing with the accumulated weight
    Pixel result;
    result.red = std::clamp(static_cast<int>(sumR / normalizationFactor), 0, 255);
    result.green = std::clamp(static_cast<int>(sumG / normalizationFactor), 0, 255);
    result.blue = std::clamp(static_cast<int>(sumB / normalizationFactor), 0, 255);
    result.alpha = 255;
    return result;
}

// Function to apply a Median Filter to the entire image
bool Denoise_Median(const char* inputFile, const char* outputFile, int kernelSize) {
    std::ifstream inFile(inputFile, std::ios::binary);
    BMPHeader bmpHeader;
    DIBHeader dibHeader;
    // Read the BMP and DIB headers
    inFile.read(reinterpret_cast<char*>(&bmpHeader), sizeof(bmpHeader));
    inFile.read(reinterpret_cast<char*>(&dibHeader), sizeof(dibHeader));

    std::cout << inputFile << " \nBit Depth: " << dibHeader.bitCount << std::endl;

    int bytesPerPixel = dibHeader.bitCount / 8;
    int width = dibHeader.width;
    int height = dibHeader.height;
    int rowSize = ((width * bytesPerPixel + 3) & ~3);

    // Load pixel data into a 2D vector
    std::vector<std::vector<Pixel>> pixels(height, std::vector<Pixel>(width));
    inFile.seekg(bmpHeader.offsetData, std::ios::beg);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Pixel pixel;
            inFile.read(reinterpret_cast<char*>(&pixel.blue), 1);
            inFile.read(reinterpret_cast<char*>(&pixel.green), 1);
            inFile.read(reinterpret_cast<char*>(&pixel.red), 1);
            if (bytesPerPixel == 4) {
                inFile.read(reinterpret_cast<char*>(&pixel.alpha), 1);
            } else {
                pixel.alpha = 255;  // For 24-bit, set alpha to opaque
            }
            pixels[y][x] = pixel;
        }
        inFile.ignore(rowSize - width * bytesPerPixel); // Skip padding bytes
    }
    inFile.close();

    // Apply the Median Filter to each pixel
    std::vector<std::vector<Pixel>> resultPixels = pixels;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            resultPixels[y][x] = MedianFilter(pixels, x, y, kernelSize);
        }
    }
    // Write the denoised image to the output file
    std::ofstream outFile(outputFile, std::ios::binary);
    outFile.write(reinterpret_cast<char*>(&bmpHeader), sizeof(bmpHeader));
    outFile.write(reinterpret_cast<char*>(&dibHeader), sizeof(dibHeader));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            outFile.write(reinterpret_cast<char*>(&resultPixels[y][x].blue), 1);
            outFile.write(reinterpret_cast<char*>(&resultPixels[y][x].green), 1);
            outFile.write(reinterpret_cast<char*>(&resultPixels[y][x].red), 1);
            if (bytesPerPixel == 4) {
                outFile.write(reinterpret_cast<char*>(&resultPixels[y][x].alpha), 1);
            }
        }
        outFile.write("\0\0\0\0", rowSize - width * bytesPerPixel);
    }
    outFile.close();

    return true;
}

// Function to apply a Bilateral Filter to the entire image
bool Denoise_Bilateral(const char* inputFile, const char* outputFile, int kernelSize, float sigmaSpatial, float sigmaColor) {
    std::ifstream inFile(inputFile, std::ios::binary);
    BMPHeader bmpHeader;
    DIBHeader dibHeader;

    inFile.read(reinterpret_cast<char*>(&bmpHeader), sizeof(bmpHeader));
    inFile.read(reinterpret_cast<char*>(&dibHeader), sizeof(dibHeader));

    std::cout << inputFile << " \nBit Depth: " << dibHeader.bitCount << std::endl;

    int bytesPerPixel = dibHeader.bitCount / 8;
    int width = dibHeader.width;
    int height = dibHeader.height;
    int rowSize = ((width * bytesPerPixel + 3) & ~3);

    std::vector<std::vector<Pixel>> pixels(height, std::vector<Pixel>(width));
    inFile.seekg(bmpHeader.offsetData, std::ios::beg);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Pixel pixel;
            inFile.read(reinterpret_cast<char*>(&pixel.blue), 1);
            inFile.read(reinterpret_cast<char*>(&pixel.green), 1);
            inFile.read(reinterpret_cast<char*>(&pixel.red), 1);
            if (bytesPerPixel == 4) {
                inFile.read(reinterpret_cast<char*>(&pixel.alpha), 1);
            } else {
                pixel.alpha = 255;
            }
            pixels[y][x] = pixel;
        }
        inFile.ignore(rowSize - width * bytesPerPixel);
    }
    inFile.close();

    std::vector<std::vector<Pixel>> resultPixels = pixels;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            resultPixels[y][x] = BilateralFilter(pixels, x, y, kernelSize, sigmaSpatial, sigmaColor);
        }
    }

    std::ofstream outFile(outputFile, std::ios::binary);
    outFile.write(reinterpret_cast<char*>(&bmpHeader), sizeof(bmpHeader));
    outFile.write(reinterpret_cast<char*>(&dibHeader), sizeof(dibHeader));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            outFile.write(reinterpret_cast<char*>(&resultPixels[y][x].blue), 1);
            outFile.write(reinterpret_cast<char*>(&resultPixels[y][x].green), 1);
            outFile.write(reinterpret_cast<char*>(&resultPixels[y][x].red), 1);
            if (bytesPerPixel == 4) {
                outFile.write(reinterpret_cast<char*>(&resultPixels[y][x].alpha), 1);
            }
        }
        outFile.write("\0\0\0\0", rowSize - width * bytesPerPixel);
    }
    outFile.close();

    return true;
}

std::string OutputName(const std::string &inputFile, int level) {
    std::regex re("input(\\d+)\\.bmp");
    std::smatch match;
    if (std::regex_search(inputFile, match, re) && match.size() > 1) {
        return "output" + match.str(1) + "_" + std::to_string(level) + ".bmp";
    }
    return "output_" + std::to_string(level) + ".bmp";
}

int main() {
    // Parameters for Median Filter
    int kernelSize1 = 3;
    int kernelSize2 = 5;

    // Parameters for Bilateral Filter
    int kernelSize3 = 9;
    float sigmaSpatial1 = 10.0, sigmaColor1 = 100.0;
    int kernelSize4 = 7;
    float sigmaSpatial2 = 8.0, sigmaColor2 = 60.0;

    // Apply Median Filter
    std::string inputFile1 = "input3.bmp";
    std::string outputFile1_median = OutputName(inputFile1, 1);
    std::string outputFile2_median = OutputName(inputFile1, 2);
    Denoise_Median(inputFile1.c_str(), outputFile1_median.c_str(), kernelSize1);
    Denoise_Median(inputFile1.c_str(), outputFile2_median.c_str(), kernelSize2);

    // Apply Bilateral Filter
    std::string inputFile2 = "input4.bmp";
    std::string outputFile1_bilateral = OutputName(inputFile2, 1);
    std::string outputFile2_bilateral = OutputName(inputFile2, 2);
    Denoise_Bilateral(inputFile2.c_str(), outputFile1_bilateral.c_str(), kernelSize3, sigmaSpatial1, sigmaColor1);
    Denoise_Bilateral(inputFile2.c_str(), outputFile2_bilateral.c_str(), kernelSize4, sigmaSpatial2, sigmaColor2);

    return 0;
}
