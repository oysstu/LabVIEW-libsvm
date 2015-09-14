CALL "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\vcvars32.bat"
SET TARGET="windows_x86"

del /s /q /f *.obj
del /s /q /f %TARGET%\*.lib

if not exist .\%TARGET% mkdir %TARGET%

CL.exe /MD /c /GL /EHsc /D _WIN32 /D _CRT_SECURE_NO_DEPRECATE /O2 linear.cpp tron.cpp blas\*.c
LIB.exe /LTCG /NOLOGO /OUT:%TARGET%\liblinear.lib *.obj