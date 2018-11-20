package rogerallen.jmandelbrotr;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUgraphicsResource;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaGLDeviceList;
import jcuda.runtime.cudaGraphicsRegisterFlags;
import jcuda.runtime.cudaGraphicsResource;

public class AppCUDA {

	public static double centerX, centerY, zoom;
	public static int iterMult;
	public static boolean doublePrecision;
	
	public static cudaGraphicsResource cudaPBOHandle = new cudaGraphicsResource();
	private static CUfunction mandelbrotFloatKernel, mandelbrotDoubleKernel;

	// return true when there is an error
	public static boolean init() {

		centerX = -0.5; 
		centerY = 0.0;
		zoom = 0.5;
		iterMult = 1;
		doublePrecision = false;

		int err;
		
		// find the first GL device & use that.
		int[] deviceCounts = {-1};
		int[] devices = {-1};
		if ((err = JCuda.cudaGLGetDevices(deviceCounts, devices, 1, cudaGLDeviceList.cudaGLDeviceListAll)) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + errStr(err) + ") failed to cudaGLGetDevices");
			return true;
		}
		if(deviceCounts[0] == 0) {
			System.err.println("ERROR: no cudaGLGetDevices found.");
			return true;			
		}
		if ((err = JCuda.cudaSetDevice(devices[0])) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + errStr(err) + ") failed to cudaSetDevice("+devices[0]+")");
			return true;
		}
		// CUDA writes to the buffer, OpenGL reads, then this repeats.
		// So, add WriteDiscard flag to this buffer.
		if ((err = JCuda.cudaGraphicsGLRegisterBuffer(cudaPBOHandle, AppGL.sharedBufID,
				cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard)) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + errStr(err) + ") Failed to register buffer " + AppGL.sharedBufID);
			System.err.println("Make sure that you are running graphics on NVIDIA GPU");
			return true;
		}

		System.out.println("Loading cuda kernels...");
		CUmodule module = new CUmodule();
		String ptxPath = "src/main/resources/mandelbrot.ptx";
		if ((err = JCudaDriver.cuModuleLoad(module, ptxPath)) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + errStr(err) + ") failed to find " + ptxPath);
			return true;
		}
		String curFunction = "mandel_float";
		mandelbrotFloatKernel = new CUfunction();
		if ((err = JCudaDriver.cuModuleGetFunction(mandelbrotFloatKernel, module,
				curFunction)) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + errStr(err) + ") failed to get function " + curFunction);
			return true;
		}
		curFunction = "mandel_double";
		mandelbrotDoubleKernel = new CUfunction();
		if ((err = JCudaDriver.cuModuleGetFunction(mandelbrotDoubleKernel, module,
				curFunction)) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + errStr(err) + ") failed to get function " + curFunction);
			return true;
		}
		return false;
	}

	public static void render() {
		CUdeviceptr devPtr = mapResouce(cudaPBOHandle);
		mandelbrot(devPtr, AppGL.sharedTexWidth, AppGL.sharedTexHeight, centerX, centerY, zoom, iterMult,
				doublePrecision);
		unmapResouce(cudaPBOHandle);
	}
	
	private static String errStr(int err) {
		return JCuda.cudaGetErrorName(err) + "=" + err;
	}

	private static CUdeviceptr mapResouce(cudaGraphicsResource cudaResource) {
		CUgraphicsResource cuResource = new CUgraphicsResource(cudaResource);
		CUdeviceptr basePointer = new CUdeviceptr();
		int err;
		if ((err = JCudaDriver.cuGraphicsMapResources(1, new CUgraphicsResource[] { cuResource },
				null)) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + errStr(err) + ") failed to map resource");
		}
		if ((err = JCudaDriver.cuGraphicsResourceGetMappedPointer(basePointer, new long[1],
				cuResource)) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + errStr(err) + ") failed to get mapped pointer");
		}
		return basePointer;
	}

	private static void unmapResouce(cudaGraphicsResource cudaResource) {
		CUgraphicsResource cuResource = new CUgraphicsResource(cudaResource);
		int err;
		if ((err = JCudaDriver.cuGraphicsUnmapResources(1, new CUgraphicsResource[] { cuResource },
				null)) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + errStr(err) + ") failed to unmap resource");
		}
		;
	}

	private static void mandelbrot(CUdeviceptr devPtr, int w, int h, double cx, double cy, double zoom, int iter,
			boolean doublePrec) {
		int blockSize = 16; // 256 threads per block
		int err;
		if (doublePrec) {
			Pointer doubleParams = Pointer.to(Pointer.to(devPtr), Pointer.to(new int[] { w }),
					Pointer.to(new int[] { h }), Pointer.to(new double[] { cx }), Pointer.to(new double[] { cy }),
					Pointer.to(new double[] { zoom }), Pointer.to(new int[] { iter }));
			if ((err = JCudaDriver.cuLaunchKernel(mandelbrotDoubleKernel, w / blockSize, h / blockSize, 1, // grids
					blockSize, blockSize, 1, // block
					0, null, // shared memory, stream
					doubleParams, null // params, extra
			)) != cudaError.cudaSuccess) {
				System.err.println("ERROR: (" + errStr(err) + ") in cuLaunchKernel for double_kernel");
			}
		} else {
			Pointer floatParams = Pointer.to(Pointer.to(devPtr), Pointer.to(new int[] { w }),
					Pointer.to(new int[] { h }), Pointer.to(new float[] { (float) cx }),
					Pointer.to(new float[] { (float) cy }), Pointer.to(new float[] { (float) zoom }),
					Pointer.to(new int[] { iter }));
			if ((err = JCudaDriver.cuLaunchKernel(mandelbrotFloatKernel, w / blockSize, h / blockSize, 1, blockSize,
					blockSize, 1, 0, null, floatParams, null)) != cudaError.cudaSuccess) {
				System.err.println("ERROR: (" + errStr(err) + ") in cuLaunchKernel for float_kernel");
			}
		}
		if ((err = JCuda.cudaGetLastError()) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + errStr(err) + ") from cudaGetLastError");
		}
		if ((err = JCudaDriver.cuCtxSynchronize()) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + errStr(err) + ") in cuCtxSynchronize");
		}
	}
}
