IF EXIST "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64\vcvars64.bat" (
	echo "Using native 64-bit compiler"
	CALL "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64\vcvars64.bat"
) ELSE (
	echo "Using 32-bit compiler to compile x64 target
	CALL "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64\vcvarsx86_amd64.bat"
)

SET TARGET="windows_x86_64"
SET CFLAGS="/nologo /O2 /EHsc /I. /D _WIN32 /D _WIN64 /D _CRT_SECURE_NO_DEPRECATE"

if not exist .\%TARGET% mkdir %TARGET%

nmake /E -f Makefile.win clean all lib

cd .\%TARGET%\
ren libsvm.dll libsvm-dense.dll
ren libsvm.exp libsvm-dense.exp
ren libsvm.lib libsvm-dense.lib
cd ..