package rogerallen.jmandelbrotr;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUgraphicsResource;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaGraphicsRegisterFlags;
import jcuda.runtime.cudaGraphicsResource;

public class AppCUDA {

	public static cudaGraphicsResource cudaPBOHandle = new cudaGraphicsResource();
	public static double centerX, centerY, zoom;
	public static int iterMult;
	public static boolean doublePrecision;

	private static CUfunction mandelbrotFloatKernel, mandelbrotDoubleKernel;

	public static void init() {

		centerX = -0.5; 
		centerY = 0.0;
		zoom = 0.5;
		iterMult = 1;
		doublePrecision = false;

		int err;
		// FIXME -- just use commandline to select the device or default to 0
		cudaDeviceProp prop = new cudaDeviceProp();
		int[] dev = { 0 };
		prop.major = 6;
		prop.minor = 0;
		if ((err = JCuda.cudaChooseDevice(dev, prop)) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + err + ") failed to choose CUDA device");
		}
		System.out.println("CUDA chose device " + dev[0]);
		if ((err = JCuda.cudaGLSetGLDevice(dev[0])) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + err + ") failed to set CUDA GL device");
		}
		if ((err = JCuda.cudaGraphicsGLRegisterBuffer(cudaPBOHandle, AppGL.sharedBufID,
				cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsNone)) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + err + ") Failed to register buffer " + AppGL.sharedBufID);
		}

		System.out.println("Loading cuda kernels...");
		CUmodule module = new CUmodule();
		String ptxPath = "src/main/resources/mandelbrot.ptx";
		if ((err = JCudaDriver.cuModuleLoad(module, ptxPath)) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + err + ") failed to find " + ptxPath);
		}
		String curFunction = "mandel_float";
		mandelbrotFloatKernel = new CUfunction();
		if ((err = JCudaDriver.cuModuleGetFunction(mandelbrotFloatKernel, module,
				curFunction)) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + err + ") failed to get function " + curFunction);
		}
		curFunction = "mandel_double";
		mandelbrotDoubleKernel = new CUfunction();
		if ((err = JCudaDriver.cuModuleGetFunction(mandelbrotDoubleKernel, module,
				curFunction)) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + err + ") failed to get function " + curFunction);
		}

	}

	public static void render() {
		CUdeviceptr devPtr = mapResouce(cudaPBOHandle);
		mandelbrot(devPtr, AppGL.sharedTexWidth, AppGL.sharedTexHeight, centerX, centerY, zoom, iterMult,
				doublePrecision);
		unmapResouce(cudaPBOHandle);
	}

	private static CUdeviceptr mapResouce(cudaGraphicsResource cudaResource) {
		CUgraphicsResource cuResource = new CUgraphicsResource(cudaResource);
		CUdeviceptr basePointer = new CUdeviceptr();
		int err;
		if ((err = JCudaDriver.cuGraphicsMapResources(1, new CUgraphicsResource[] { cuResource },
				null)) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + err + ") failed to map resource");
		}
		if ((err = JCudaDriver.cuGraphicsResourceGetMappedPointer(basePointer, new long[1],
				cuResource)) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + err + ") failed to get mapped pointer");
		}
		return basePointer;
	}

	private static void unmapResouce(cudaGraphicsResource cudaResource) {
		CUgraphicsResource cuResource = new CUgraphicsResource(cudaResource);
		int err;
		if ((err = JCudaDriver.cuGraphicsUnmapResources(1, new CUgraphicsResource[] { cuResource },
				null)) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + err + ") failed to unmap resource");
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
				System.err.println("ERROR: (" + err + ") in cuLaunchKernel for double_kernel");
			}
		} else {
			Pointer floatParams = Pointer.to(Pointer.to(devPtr), Pointer.to(new int[] { w }),
					Pointer.to(new int[] { h }), Pointer.to(new float[] { (float) cx }),
					Pointer.to(new float[] { (float) cy }), Pointer.to(new float[] { (float) zoom }),
					Pointer.to(new int[] { iter }));
			if ((err = JCudaDriver.cuLaunchKernel(mandelbrotFloatKernel, w / blockSize, h / blockSize, 1, blockSize,
					blockSize, 1, 0, null, floatParams, null)) != cudaError.cudaSuccess) {
				System.err.println("ERROR: (" + err + ") in cuLaunchKernel for float_kernel");
			}
		}
		if ((err = JCuda.cudaGetLastError()) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + err + ") from cudaGetLastError");
		}
		if ((err = JCudaDriver.cuCtxSynchronize()) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + err + ") in cuCtxSynchronize");
		}
	}
}
