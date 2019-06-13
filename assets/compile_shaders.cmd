@echo off

setlocal
set GLSL_COMPILER=glslangValidator.exe
set SOURCE_FOLDER="./../vkFlow/shaders/"
set BINARIES_FOLDER="./shaders/"

:: kernel shaders
%GLSL_COMPILER% -V -S comp %SOURCE_FOLDER%linear_layer.glsl -o %BINARIES_FOLDER%linear_layer.bin

pause
