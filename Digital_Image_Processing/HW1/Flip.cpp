#include <iostream> // for input/output operations
#include <fstream> // for file handling (reading/writing files)
#include <vector> // for using the vector container to hold pixel data
#include <cstdint> // for using fixed-width integer types
#include <string> // for using strings to handle file names
#include <sstream> // for generating output file names

// structure definitions to hold BMP headers
#pragma pack(push, 1)
struct FileHeader {
    uint16_t bfType;      // file type "BM"
    uint32_t bfSize;      // size
    uint16_t bfReserved1; // reserved, must be 0
    uint16_t bfReserved2; // reserved, must be 0
    uint32_t bfOffBits;   // offset to start of pixel data
};

struct InfoHeader {
    uint32_t biSize;          // size of this header, typically 40 bytes
    int32_t biWidth;          // width of the image in pixels
    int32_t biHeight;         // height of the image in pixels
    uint16_t biPlanes;        // color planes number (must be 1)
    uint16_t biBitCount;      // number of bits per pixel (24, 32)
    uint32_t biCompression;   // compression type (0 = no compression)
    uint32_t biSizeImage;     // image size
    int32_t biXPelsPerMeter;  // horizontal resolution in pixels per meter
    int32_t biYPelsPerMeter;  // vertical resolution in pixels per meter
    uint32_t biClrUsed;       // colors number in the color palette (0 means all colors used)
    uint32_t biClrImportant;  // important colors number (0 means all colors are important)
};
#pragma pack(pop)

// flip image horizontally (24, 32-bit depth)
// `pixelData` is the raw image data, and `rowSize` is the size of each row (including padding)
void pixel_swap(std::vector<uint8_t> &pixelData, int width, int height, int bytesPerPixel, int rowSize) {
    // iterate over each row in the image
    for (int y = 0; y < height; ++y) {
        // swap the pixels in each row to flip the image horizontally
        for (int x = 0; x < width / 2; ++x) {
            int leftIndex = y * rowSize + x * bytesPerPixel; // left pixel index
            int rightIndex = y * rowSize + (width - 1 - x) * bytesPerPixel; // right pixel index
            // swap the left and right pixel values
            for (int i = 0; i < bytesPerPixel; ++i) {
                std::swap(pixelData[leftIndex + i], pixelData[rightIndex + i]);
            }
        }
    }
}

// read a BMP file, flip it horizontally, and save the result as a new BMP file
void flip_horizontal(const std::string &inputFileName, const std::string &outputFileName) {
    // open the input BMP file in binary mode
    std::ifstream inputFile(inputFileName, std::ios::binary);
    // open the output BMP file in binary mode
    std::ofstream outputFile(outputFileName, std::ios::binary);

    FileHeader fileHeader; // BMP file header (contains metadata about the BMP file)
    InfoHeader infoHeader; // BMP info header (contains information about the image)
    
    // read the BMP file and info headers from the input file
    inputFile.read(reinterpret_cast<char *>(&fileHeader), sizeof(fileHeader));
    inputFile.read(reinterpret_cast<char *>(&infoHeader), sizeof(infoHeader));

    int width = infoHeader.biWidth; // image width
    int height = infoHeader.biHeight; // image height
    int bitDepth = infoHeader.biBitCount; // bits per pixel (24 or 32)

std::cout << "Original Image: " << inputFileName << " \nBit Depth: " << bitDepth << std::endl;

    // calculate the size of each row in bytes (taking into account any padding)
    int rowSize = ((bitDepth * width + 31) / 32) * 4;
    // total size of the image data in bytes
    int dataSize = rowSize * height;
    // vector to hold the pixel data
    std::vector<uint8_t> pixelData(dataSize); 

    // seek to the start of the pixel data (skip the headers)
    inputFile.seekg(fileHeader.bfOffBits, std::ios::beg);
    // read the pixel data into the vector
    inputFile.read(reinterpret_cast<char *>(pixelData.data()), dataSize);

    // flip the image horizontally for 24-bit and 32-bit BMPs
    if (bitDepth == 24 || bitDepth == 32) {
        int bytesPerPixel = bitDepth / 8; // number of bytes per pixel (3 for 24-bit, 4 for 32-bit)
        pixel_swap(pixelData, width, height, bytesPerPixel, rowSize);
    } 

    // write the BMP headers to the output file
    outputFile.write(reinterpret_cast<char *>(&fileHeader), sizeof(fileHeader));
    outputFile.write(reinterpret_cast<char *>(&infoHeader), sizeof(infoHeader));
    // write the flipped pixel data to the output file
    outputFile.write(reinterpret_cast<char *>(pixelData.data()), dataSize);
    // close the input and output files
    inputFile.close();
    outputFile.close();
}

// generate an output file name by appending a number to "output_flip.bmp"
std::string OutputFile(const std::string &inputFileName, int index) {
    std::stringstream ss;
    ss << "output" << index + 1 << "_flip.bmp";
    return ss.str();
}

// main function to process multiple BMP images
int main() {
    std::vector<std::string> inputFileNames = {
        "input1.bmp", "input2.bmp"  // add as many input file names as needed
    };

    // loop through the list of BMP files
    for (size_t i = 0; i < inputFileNames.size(); ++i) {
        // generate a unique output file name for each image
        std::string outputFileName = OutputFile(inputFileNames[i], i);
        // flip the image and save it to the output file
        flip_horizontal(inputFileNames[i], outputFileName);
    }

    return 0;
}
