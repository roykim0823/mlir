
mkdir -p build
llc -filetype=obj --relocation-model=pic simple.ll -o ./build/simple.o
clang -shared -fPIC ./build/simple.o -o ./build/libsimple.so
clang ./build/simple.o -o ./build/simple # optionally create an executable
./build/simple; echo $?

python3 simple.py