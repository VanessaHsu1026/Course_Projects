#include <iostream> // for input-output operations
#include <fstream> // for file handling (reading/writing files)
#include <vector> // for handling dynamic arrays
#include <cstdint> // for using fixed-width integer types

// ensure no padding between structure members, ensuring compatibility with BMP file format
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
    int32_t biHeight;         // height of the image in pixels (can be negative for top-down images)
    uint16_t biPlanes;        // color planes number (must be 1)
    uint16_t biBitCount;      // number of bits per pixel
    uint32_t biCompression;   // compression type (0 = no compression)
    uint32_t biSizeImage;     // size of image data
    int32_t biXPelsPerMeter;  // horizontal resolution in pixels per meter
    int32_t biYPelsPerMeter;  // vertical resolution in pixels per meter
    uint32_t biClrUsed;       // number of colors in the palette (0 means default)
    uint32_t biClrImportant;  // important color count (0 means all colors are important)
};
#pragma pack(pop)

void cropping(const std::string &inputFileName, const std::string &outputFileName, int cropX, int cropY, int cropWidth, int cropHeight) {
    // open input BMP file in binary mode
    std::ifstream inputFile(inputFileName, std::ios::binary);
    // open output BMP file in binary mode
    std::ofstream outputFile(outputFileName, std::ios::binary);

    FileHeader fileHeader;
    InfoHeader infoHeader;

    // read BMP headers from the input file
    inputFile.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
    inputFile.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));

    int originalWidth = infoHeader.biWidth;
    int originalHeight = infoHeader.biHeight;
    int bitDepth = infoHeader.biBitCount;

    std::cout << "Original Image: " << inputFileName << "\nWidth: " << originalWidth << " Height: " << originalHeight << " Bit Depth: " << bitDepth << std::endl;

    // adjust crop region to fit within the image dimensions
    if (cropX + cropWidth > originalWidth) cropWidth = originalWidth - cropX;
    if (cropY + cropHeight > originalHeight) cropHeight = originalHeight - cropY;

    // calculate row size (padded to a multiple of 4 bytes)
    int rowSize = ((bitDepth * originalWidth + 31) / 32) * 4; // original row size with padding
    int croppedRowSize = ((bitDepth * cropWidth + 31) / 32) * 4; // cropped row size with padding

    // update the headers for the cropped image
    infoHeader.biWidth = cropWidth; // new width for the cropped image
    infoHeader.biHeight = cropHeight; // new height for the cropped image
    infoHeader.biSizeImage = croppedRowSize * cropHeight; // new image size for the cropped image
    fileHeader.bfSize = fileHeader.bfOffBits + infoHeader.biSizeImage; // new file size

    // write updated headers to the output file
    outputFile.write(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
    outputFile.write(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));

    // buffer to hold each row of the original image
    std::vector<uint8_t> rowData(rowSize);
    // buffer to hold each row of the cropped image
    std::vector<uint8_t> croppedRowData(croppedRowSize);

    // loop through each row of the original image
    for (int y = 0; y < originalHeight; ++y) {
        // read one row of pixel data
        inputFile.read(reinterpret_cast<char*>(rowData.data()), rowSize);
        // if the row is within the crop region
        if (y >= cropY && y < cropY + cropHeight) {
            if (bitDepth == 24) { // if it's a 24-bit BMP (3 bytes per pixel)
                std::copy(rowData.begin() + cropX * 3, rowData.begin() + (cropX + cropWidth) * 3, croppedRowData.begin());
            } else if (bitDepth == 32) { // if it's a 32-bit BMP (4 bytes per pixel)
                std::copy(rowData.begin() + cropX * 4, rowData.begin() + (cropX + cropWidth) * 4, croppedRowData.begin());
            }
            outputFile.write(reinterpret_cast<char*>(croppedRowData.data()), croppedRowSize); // write the cropped row
        }
    }

    inputFile.close(); // close the input file
    outputFile.close(); // close the output file
}

int main() {
    // crop parameters for the first image (input1.bmp)
    int cropX1 = 120; // starting x-coordinate of the crop
    int cropY1 = 150; // starting y-coordinate of the crop
    int cropWidth1 = 400; // width of the crop
    int cropHeight1 = 399; // height of the crop

    // crop parameters for the second image (input2.bmp)
    int cropX2 = 120; // starting x-coordinate of the crop
    int cropY2 = 150; // starting y-coordinate of the crop
    int cropWidth2 = 100; // width of the crop
    int cropHeight2 = 100; // height of the crop

    // crop both images and save
    // call any number of input file names as needed
    cropping("input1.bmp", "output1_crop.bmp", cropX1, cropY1, cropWidth1, cropHeight1);
    cropping("input2.bmp", "output2_crop.bmp", cropX2, cropY2, cropWidth2, cropHeight2);

    return 0;
}
