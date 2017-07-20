IF EXIST "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat" (
	echo "Using native 64-bit compiler"
	CALL "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
) ELSE (
	echo "Using 32-bit compiler to compile x64 target
	CALL "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsx86_amd64.bat"
)

SET TARGET="windows_x86_64"

del /s /q /f *.obj
del /s /q /f %TARGET%\*.lib

if not exist .\%TARGET% mkdir %TARGET%

CL.exe /GL /c /EHsc /MD /D _WIN64 /D _CRT_SECURE_NO_DEPRECATE /D _DENSE_REP /O2 svm.cpp
LIB.exe /LTCG /NOLOGO /OUT:%TARGET%\libsvm-dense.lib svm.obj