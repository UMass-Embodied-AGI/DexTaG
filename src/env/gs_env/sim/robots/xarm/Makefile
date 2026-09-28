all: libxarm7_kinematics.so

libxarm7_kinematics.so: xarm7_ctypes_wrapper.cpp libkinematics_xarm_user.a
	g++ -std=c++11 -shared -fPIC -O2 xarm7_ctypes_wrapper.cpp libkinematics_xarm_user.a -o $@ -fopenmp

clean:
	rm -f libxarm7_kinematics.so
