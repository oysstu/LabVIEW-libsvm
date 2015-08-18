CALL "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\vcvars32.bat"
SET TARGET="windows_x86"

if not exist .\%TARGET% mkdir %TARGET%

nmake /E -f Makefile.win clean all lib