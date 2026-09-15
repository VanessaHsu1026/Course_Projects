#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <string>
#include <regex>
#include <algorithm>

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
    uint16_t bitCount{0}; // 24-bit or 32-bit
    uint32_t compression{0};
    uint32_t imageSize{0};
    int32_t xPixelsPerMeter{0};
    int32_t yPixelsPerMeter{0};
    uint32_t colorsUsed{0};
    uint32_t colorsImportant{0};
};
#pragma pack(pop)

struct Pixel {
    uint8_t blue;   // Blue component
    uint8_t green;  // Green component
    uint8_t red;    // Red component
    uint8_t alpha;  // Alpha channel (only used for 32-bit BMP)
};

// Function to apply a sharpening kernel to a specific pixel
Pixel Kernel(const std::vector<std::vector<Pixel>>& pixels, int x, int y, const std::vector<std::vector<int>>& kernel, int bytesPerPixel) {
    int kernelSize = kernel.size();
    int halfKernel = kernelSize / 2;
    int sumR = 0, sumG = 0, sumB = 0;

    // Iterate over the kernel
    for (int ky = -halfKernel; ky <= halfKernel; ++ky) {
        for (int kx = -halfKernel; kx <= halfKernel; ++kx) {
            // Get the pixel at the corresponding position, clamped to image boundaries
            int px = std::clamp(x + kx, 0, (int)pixels[0].size() - 1);
            int py = std::clamp(y + ky, 0, (int)pixels.size() - 1);
            Pixel p = pixels[py][px];
            int kValue = kernel[ky + halfKernel][kx + halfKernel];

            // Accumulate weighted RGB values based on kernel
            sumR += p.red * kValue;
            sumG += p.green * kValue;
            sumB += p.blue * kValue;
        }
    }
    // Create the result pixel with clamped RGB values
    Pixel result;
    result.red = std::clamp(sumR, 0, 255);
    result.green = std::clamp(sumG, 0, 255);
    result.blue = std::clamp(sumB, 0, 255);
    if (bytesPerPixel == 4) {
        result.alpha = pixels[y][x].alpha;  // Preserve the alpha channel for 32-bit BMP
    }

    return result;
}

// Function to apply sharpness to an image by using a convolution kernel
bool Sharpness(const char* inputFile, const char* outputFile, const std::vector<std::vector<int>>& kernel) {
    // Open the BMP file for reading in binary mode
    std::ifstream inFile(inputFile, std::ios::binary);
    BMPHeader bmpHeader;
    DIBHeader dibHeader;
    // Read BMP and DIB headers
    inFile.read(reinterpret_cast<char*>(&bmpHeader), sizeof(bmpHeader));
    inFile.read(reinterpret_cast<char*>(&dibHeader), sizeof(dibHeader));

    std::cout << inputFile << " \nBit Depth: " << dibHeader.bitCount << std::endl;
    
    // Calculate bytes per pixel (3 for RGB, 4 for RGBA)
    int bytesPerPixel = dibHeader.bitCount / 8;

    int width = dibHeader.width;
    int height = dibHeader.height;
    int rowSize = ((width * bytesPerPixel + 3) & ~3);  // Row size with padding
    
    // Load pixel data from BMP into a 2D vector
    std::vector<std::vector<Pixel>> pixels(height, std::vector<Pixel>(width));
    inFile.seekg(bmpHeader.offsetData, std::ios::beg); // Move to pixel data offset
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Pixel pixel;
            inFile.read(reinterpret_cast<char*>(&pixel.blue), 1);
            inFile.read(reinterpret_cast<char*>(&pixel.green), 1);
            inFile.read(reinterpret_cast<char*>(&pixel.red), 1);
            if (bytesPerPixel == 4) { // Read alpha channel if present
                inFile.read(reinterpret_cast<char*>(&pixel.alpha), 1);
            }
            pixels[y][x] = pixel;
        }
        inFile.ignore(rowSize - width * bytesPerPixel);  // Skip padding bytes
    }
    inFile.close();

    // Apply the kernel to each pixel
    std::vector<std::vector<Pixel>> resultPixels = pixels;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            resultPixels[y][x] = Kernel(pixels, x, y, kernel, bytesPerPixel);
        }
    }

    // Write the processed image to a new BMP file
    std::ofstream outFile(outputFile, std::ios::binary);
    outFile.write(reinterpret_cast<char*>(&bmpHeader), sizeof(bmpHeader));
    outFile.write(reinterpret_cast<char*>(&dibHeader), sizeof(dibHeader));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            outFile.write(reinterpret_cast<char*>(&resultPixels[y][x].blue), 1);
            outFile.write(reinterpret_cast<char*>(&resultPixels[y][x].green), 1);
            outFile.write(reinterpret_cast<char*>(&resultPixels[y][x].red), 1);
            if (bytesPerPixel == 4) { // Write alpha channel if present
                outFile.write(reinterpret_cast<char*>(&resultPixels[y][x].alpha), 1);
            }
        }
        outFile.write("\0\0\0\0", rowSize - width * bytesPerPixel);  // Add padding bytes if needed
    }
    outFile.close();

    return true;
}

// Generate an output filename based on the input filename and sharpening level
std::string OutputName(const std::string &inputFile, int level) {
    std::regex re("input(\\d+)\\.bmp");
    std::smatch match;
    // Check if input filename matches the pattern
    if (std::regex_search(inputFile, match, re) && match.size() > 1) {
        // Append the sharpening level to the output filename
        return "output" + match.str(1) + "_" + std::to_string(level) + ".bmp";
    }
    // Default output filename if input pattern does not match
    return "output_" + std::to_string(level) + ".bmp";
}

// Main function
int main() {
    std::vector<std::string> inputFiles = {"input2.bmp"}; // List of input BMP files
    
    // Strong Sharpening Kernel
    std::vector<std::vector<int>> strong_kernel = {
        {-1, -1, -1},
        {-1, 9, -1},
        {-1, -1, -1}
    };

    // Unsharp Masking Kernel
    std::vector<std::vector<int>> unsharp_kernel = {
        {-1, -2, -1},
        {-2, 13, -2},
        {-1, -2, -1}
    };

    // Apply both sharpening kernels to each input file
    for (const auto& inputFile : inputFiles) {
        std::string OutputFile1 = OutputName(inputFile, 1); // Output file for strong kernel
        std::string OutputFile2 = OutputName(inputFile, 2); // Output file for unsharp kernel

        Sharpness(inputFile.c_str(), OutputFile1.c_str(), strong_kernel); // Apply strong kernel
        Sharpness(inputFile.c_str(), OutputFile2.c_str(), unsharp_kernel); // Apply unsharp kernel
        
    }

    return 0;
}
