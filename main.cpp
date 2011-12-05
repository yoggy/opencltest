//
//	OpenCL test program (for Mac OS X)
//
//	$ cmake .
//	$ make
//	$ ./opencltest
//
#include "cl.hpp"
#include <string>
#include <vector>
#include <iostream>

// for opencv
cl::Platform platform_;
cl::Context context_;
cl::Device device_;
cl::Program program_;
cl::Program::Sources sources_;
cl::Buffer buf_;
cl::CommandQueue queue_;
cl::Kernel kernel_;

std::string source_str = "__kernel void hello(__global *in){}";

#define CHECK_CL_ERROR(err, str) {if(err!=CL_SUCCESS){std::cerr<<"ERROR : "<<str<<"("<<err<<")"<<std::endl; return -1;}}

int main(int argc, char *argv[])
{
	cl_int err = CL_SUCCESS;
	cl::Event evt;

	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		return false;
	}
	platform_ = platforms[0];

	cl_context_properties properties[] = 
		{ CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
	context_ = cl::Context(CL_DEVICE_TYPE_GPU, properties, NULL, NULL, &err); 
	CHECK_CL_ERROR(err, "cl::Context");

	std::vector<cl::Device> devices = context_.getInfo<CL_CONTEXT_DEVICES>();
	if (devices.size() == 0) {
		return false;
	}
	device_ = devices[0];

	sources_.push_back(std::make_pair(source_str.c_str(), source_str.size()));
	program_ = cl::Program(context_, sources_);
	err = program_.build(devices);
	if (err != CL_SUCCESS) {
		std::string log = program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
		std::cout << "program.build() ERROR: " << log.c_str() << std::endl;
		return false;
	}

	kernel_ = cl::Kernel(program_, "hello", &err); 
	CHECK_CL_ERROR(err, "cl::Kernel");

	buf_ = cl::Buffer(context_, CL_MEM_READ_ONLY, 1024, NULL, &err);

	queue_ = cl::CommandQueue(context_, device_, 0, &err);
	CHECK_CL_ERROR(err, "cl::CommandQueue");

	kernel_.setArg(0, buf_);

	err = queue_.enqueueNDRangeKernel(kernel_, cl::NullRange, cl::NDRange(10, 10), cl::NullRange, NULL, &evt); 
	evt.wait();
	CHECK_CL_ERROR(err, "queue.enqueueNDRangeKernel()");

	return 0;
}

