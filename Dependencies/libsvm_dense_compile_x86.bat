CALL "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" x86
SET TARGET="windows_x86"

if not exist .\%TARGET% mkdir %TARGET%

nmake /E -f Makefile.win clean all lib

cd .\%TARGET%\
ren libsvm.dll libsvm-dense.dll
ren libsvm.exp libsvm-dense.exp
ren libsvm.lib libsvm-dense.lib
cd ..