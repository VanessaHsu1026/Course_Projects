#include <iostream> // for input/output operations
#include <fstream> // for file handling (reading/writing files)
#include <vector> // for using the vector container to hold pixel data
#include <cstdint> // for using fixed-width integer types
#include <string> // for using strings to handle file names
#include <sstream> // for generating output file names

// BMP file format requires specific byte alignment; the pragma ensures no padding
#pragma pack(push, 1)
struct FileHeader {
    uint16_t bfType;      // file type "BM"
    uint32_t bfSize;      // size
    uint16_t bfReserved1; // reserved, must be 0
    uint16_t bfReserved2; // reserved, must be 0
    uint32_t bfOffBits;   // offset to start of pixel data
};

struct InfoHeader {
    uint32_t biSize;          // size of this header
    int32_t biWidth;          // width of the image in pixels
    int32_t biHeight;         // height of the image in pixels
    uint16_t biPlanes;        // color planes number (must be 1)
    uint16_t biBitCount;      // number of bits per pixel (usually 24 for RGB or 32 for RGBA)
    uint32_t biCompression;   // compression type (0 = no compression)
    uint32_t biSizeImage;     // image size
    int32_t biXPelsPerMeter;  // horizontal resolution in pixels per meter
    int32_t biYPelsPerMeter;  // vertical resolution in pixels per meter
    uint32_t biClrUsed;       // colors number in the color table (0 if none)
    uint32_t biClrImportant;  // important colors number (0 means all colors are important)
};
#pragma pack(pop)

// // quantize an 8-bit value to a lower bit-depth
// uint8_t quantize_value(uint8_t value, int bits) {
//     int levels = (1 << bits) - 1;  // calculate number of quantization levels based on bits
//     return (value * levels / 255) * (255 / levels);  // scale the value to the quantized level and back to 8 bits
// }

// quantize an 8-bit value to a lower bit-depth by truncating the least significant bits
uint8_t quantize_value(uint8_t value, int bits) {
    int shift = 8 - bits; // calculate the number of bits to truncate
    return (value >> shift) << shift; // right shift to reduce bits, then left shift to pad with 0s
}

// perform BMP quantization
void quantization(const std::string &inputFileName, const std::string &outputFileName, int targetBitsPerChannel) {
    // open the input BMP file for reading in binary mode
    std::ifstream inputFile(inputFileName, std::ios::binary);
    // open the output BMP file for writing in binary mode
    std::ofstream outputFile(outputFileName, std::ios::binary);

    FileHeader fileHeader;
    InfoHeader infoHeader;

    // read the BMP file headers (file and info)
    inputFile.read(reinterpret_cast<char *>(&fileHeader), sizeof(fileHeader));
    inputFile.read(reinterpret_cast<char *>(&infoHeader), sizeof(infoHeader));

    int width = infoHeader.biWidth; // get image width
    int height = infoHeader.biHeight; // get image height
    int bitDepth = infoHeader.biBitCount; // get the bit depth (24 or 32)

std::cout << "Original Image: " << inputFileName << " \nBit Depth: " << bitDepth << std::endl;

    // calculate row size (padded to multiple of 4 bytes)
    int rowSize = ((bitDepth * width + 31) / 32) * 4;
    int dataSize = rowSize * height; // total size of pixel data

    // read pixel data into a vector
    std::vector<uint8_t> pixelData(dataSize);
    // move file pointer to the start of the pixel data
    inputFile.seekg(fileHeader.bfOffBits, std::ios::beg);
    // read pixel data
    inputFile.read(reinterpret_cast<char *>(pixelData.data()), dataSize);

    // apply quantization to 24-bit or 32-bit BMP images
    if (bitDepth == 24 || bitDepth == 32) {
        // determine whether the image has 3 (RGB) or 4 (RGBA) channels
        int bytesPerPixel = (bitDepth == 24) ? 3 : 4; 
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // get the index of the current pixel
                int pixelIndex = y * rowSize + x * bytesPerPixel;
                pixelData[pixelIndex] = quantize_value(pixelData[pixelIndex], targetBitsPerChannel);         // red
                pixelData[pixelIndex + 1] = quantize_value(pixelData[pixelIndex + 1], targetBitsPerChannel); // green
                pixelData[pixelIndex + 2] = quantize_value(pixelData[pixelIndex + 2], targetBitsPerChannel); // blue
                if (bitDepth == 32) {
                    // alpha channel remains unchanged
                }
            }
        }
    }

    // write the BMP headers back to the output BMP file
    outputFile.write(reinterpret_cast<char *>(&fileHeader), sizeof(fileHeader));
    outputFile.write(reinterpret_cast<char *>(&infoHeader), sizeof(infoHeader));
    outputFile.write(reinterpret_cast<char *>(pixelData.data()), dataSize);
    // close both input and output files
    inputFile.close();
    outputFile.close();
}

// generate output BMP file names based on input file index and quantization level
std::string OutputFile(const std::string &inputFileName, int fileIndex, int quantizationIndex) {
    std::stringstream ss;
    ss << "output" << (fileIndex + 1) << "_" << (quantizationIndex + 1) << ".bmp";
    return ss.str();
}

int main() {
    // vector to store multiple image input file names
    std::vector<std::string> inputFileNames = {"input1.bmp", "input2.bmp"}; // add as many input file names as needed

    // quantization bit levels to apply (e.g., 6 bits, 4 bits, 2 bits)
    std::vector<int> bitLevels = {6, 4, 2};

    // loop through each input file
    for (size_t i = 0; i < inputFileNames.size(); ++i) {
        // loop through each quantization level
        for (size_t j = 0; j < bitLevels.size(); ++j) {
            // generate an output file name for each quantized version of the image
            std::string outputFileName = OutputFile(inputFileNames[i], i, j);
            // quantize the image and save it to the output file
            quantization(inputFileNames[i], outputFileName, bitLevels[j]);
        }
    }

    return 0;
}
