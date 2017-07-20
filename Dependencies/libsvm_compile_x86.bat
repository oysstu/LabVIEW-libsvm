CALL "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars32.bat"

SET TARGET="windows_x86"

del /s /q /f *.obj
del /s /q /f %TARGET%\*.lib

if not exist .\%TARGET% mkdir %TARGET%

CL.exe /GL /MD /c /EHsc /D _WIN32 /D _CRT_SECURE_NO_DEPRECATE /O2 svm.cpp
LIB.exe /LTCG /NOLOGO /OUT:%TARGET%\libsvm.lib svm.obj