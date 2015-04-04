CALL "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\vcvars32.bat"
SET TARGET="windows_x86_64"
SET CFLAGS="/nologo /O2 /EHsc /I. /D _WIN32 /D _WIN64 /D _CRT_SECURE_NO_DEPRECATE"

if not exist .\%TARGET% mkdir %TARGET%

nmake /E -f Makefile.win clean all lib