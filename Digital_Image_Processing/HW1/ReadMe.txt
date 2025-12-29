Environment Setup (for Windows):
1.  Install Visual Studio Code 
2.  Click Extensions
3.  Install C/C++, C/C++ Extension Pack, C/C++ Themes 
4.  Search for "MSYS2" online and download it  
5.  Run the following command: pacman -S mingw-w64-ucrt-x86_64-gcc 
6.  "MSYS2" has been successfully installed
7.  Go to the Windows setup page
8.  Search for "Environmental Variables" 
9.  Click "Path" in the system variables and edit it 
10. Add new path: C:\msys64\mingw64\bin
11. Create a new folder named "C++" on the Windows desktop 
12. Create a folder named ".vscode" in the "C++" folder 
13. Open Visual Studio Code 
14. Select "Open Folder" 
15. Click on the "C++" folder 
16. Enter to the ".vscode" folder 
17. Add four .json files: c_cpp_properties.json, launch.json, tasks.json, settings.json 
18. The corresponding code in each .json file can be copied and pasted from the following link: https://hackmd.io/@HerryFan/bigtree 
19. Completion of environmental construction

Running the Code:
1.  Copy and paste the three C++ programs (flip, resolution, cropping) from the "hw1_113064525" folder into the "C++" folder
2.  Add the BMP input files to the "C++" folder (note that the file names of the BMP input files must be consistent with the code)
3.  Open the "C++" folder in Visual Studio Code to run the three C++ programs
4.  Press F5 to execute the code 
5.  Locate the output BMP files in the "C++" folder