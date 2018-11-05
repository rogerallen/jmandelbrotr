package rogerallen.jmandelbrotr;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUgraphicsResource;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaGraphicsResource;
import jcuda.runtime.cudaGraphicsRegisterFlags;
import jcuda.runtime.cudaDeviceProp;

public class AppCUDA {

	public static cudaGraphicsResource cuda_pbo_handle = new cudaGraphicsResource();
	public static double centerX, centerY, zoom;
	public static int iterMult;
	public static boolean doublePrecision;

	private static CUfunction mandelbrot_float_kernel, mandelbrot_double_kernel;  // FIXME snake_case

	public static void init() {

		centerX = centerY = 0.0;
		zoom = 1.0;
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
		if ((err = JCuda.cudaGraphicsGLRegisterBuffer(cuda_pbo_handle, AppGL.shared_buf_id,
				cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsNone)) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + err + ") Failed to register buffer " + AppGL.shared_buf_id);
		}

		System.out.println("Loading cuda kernels...");
		CUmodule module = new CUmodule();
		String ptx_path = "src/main/resources/mandelbrot.ptx";
		if ((err = JCudaDriver.cuModuleLoad(module, ptx_path)) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + err + ") failed to find " + ptx_path);
		}
		String cur_function = "mandel_float";
		mandelbrot_float_kernel = new CUfunction();
		if ((err = JCudaDriver.cuModuleGetFunction(mandelbrot_float_kernel, module,
				cur_function)) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + err + ") failed to get function " + cur_function);
		}
		cur_function = "mandel_double";
		mandelbrot_double_kernel = new CUfunction();
		if ((err = JCudaDriver.cuModuleGetFunction(mandelbrot_double_kernel, module,
				cur_function)) != cudaError.cudaSuccess) {
			System.err.println("ERROR: (" + err + ") failed to get function " + cur_function);
		}

	}

	public static void render() {
		CUdeviceptr devPtr = mapResouce(cuda_pbo_handle);
		mandelbrot(devPtr, AppGL.shared_tex_width, AppGL.shared_tex_height, centerX, centerY, zoom, iterMult,
				doublePrecision);
		unmapResouce(cuda_pbo_handle);
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
			if ((err = JCudaDriver.cuLaunchKernel(mandelbrot_double_kernel, w / blockSize, h / blockSize, 1, // grids
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
			if ((err = JCudaDriver.cuLaunchKernel(mandelbrot_float_kernel, w / blockSize, h / blockSize, 1, blockSize,
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
