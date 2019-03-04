package rogerallen.jmandelbrotr;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUgraphicsResource;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.nvrtc.nvrtcProgram;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaGLDeviceList;
import jcuda.runtime.cudaGraphicsRegisterFlags;
import jcuda.runtime.cudaGraphicsResource;

import static jcuda.nvrtc.JNvrtc.nvrtcCompileProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcCreateProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcGetProgramLog;
import static jcuda.nvrtc.JNvrtc.nvrtcGetPTX;
import static jcuda.nvrtc.JNvrtc.nvrtcDestroyProgram;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

public class AppCUDA {
    // realtime compile or load ptx?
    private final static boolean USE_REAL_TIME_COMPILE = true;

    private static AppWindow window;
    private static AppPbo sharedPbo;
    private static CUfunction mandelbrotFloatKernel, mandelbrotDoubleKernel;

    public static double centerX, centerY, zoom;
    public static int iterMult;
    public static boolean doublePrecision;

    // return true when there is an error
    public static boolean init(AppWindow appWindow, AppPbo sharedPbo) throws IOException {

        window = appWindow;
        AppCUDA.sharedPbo = sharedPbo;

        centerX = -0.5;
        centerY = 0.0;
        zoom = 0.5;
        iterMult = 1;
        doublePrecision = false;

        int err;

        // find the first GL device & use that.
        int[] deviceCounts = { -1 };
        int[] devices = { -1 };
        if ((err = JCuda.cudaGLGetDevices(deviceCounts, devices, 1,
                cudaGLDeviceList.cudaGLDeviceListAll)) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + errStr(err) + ") failed to cudaGLGetDevices");
            System.err.println("Make sure that you are running graphics on NVIDIA GPU");
            return true;
        }
        if (deviceCounts[0] == 0) {
            System.err.println("ERROR: no cudaGLGetDevices found.");
            return true;
        }
        if ((err = JCuda.cudaSetDevice(devices[0])) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + errStr(err) + ") failed to cudaSetDevice(" + devices[0] + ")");
            return true;
        }
        // CUDA writes to the buffer, OpenGL reads, then this repeats.
        // So, add WriteDiscard flag to this buffer.
        cudaGraphicsResource cudaPBOHandle = new cudaGraphicsResource();
        if ((err = JCuda.cudaGraphicsGLRegisterBuffer(cudaPBOHandle, sharedPbo.id(),
                cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard)) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + errStr(err) + ") Failed to register buffer " + sharedPbo.id());
            System.err.println("Make sure that you are running graphics on NVIDIA GPU");
            return true;
        }
        sharedPbo.cudaPBOHandle(cudaPBOHandle);

        CUmodule module = new CUmodule();
        if (USE_REAL_TIME_COMPILE) {
            String cudaPath = App.RESOURCES_PREFIX + "mandelbrot.cu";
            if (compileCuda(module, cudaPath)) {
                return true;
            }
        } else {
            String ptxPath = "src/main/resources/mandelbrot.ptx";
            if (loadPtx(module, ptxPath)) {
                return true;
            }
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

    private static boolean loadPtx(CUmodule module, String ptxPath) {
        int err;
        System.out.println("Loading ptx directly...");
        if ((err = JCudaDriver.cuModuleLoad(module, ptxPath)) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + errStr(err) + ") failed to find " + ptxPath);
            return true;
        }
        return false;
    }

    private static boolean compileCuda(CUmodule module, String filename) throws IOException {
        int err;
        System.out.println("Compiling cuda kernels...");
        String programSourceCode = StandardCharsets.UTF_8.decode(AppUtils.ioResourceToByteBuffer(filename)).toString();
        // Use the NVRTC to create a program by compiling the source code
        nvrtcProgram program = new nvrtcProgram();
        if ((err = nvrtcCreateProgram(program, programSourceCode, null, 0, null, null)) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + errStr(err) + ") Unable to nvrtcCreateProgram");
            return true;
        }
        cudaDeviceProp devProp = new cudaDeviceProp();
        if ((err = JCuda.cudaGetDeviceProperties(devProp, 0)) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + errStr(err) + ") Unable to cudaGetDeviceProperties");
            return true;
        }
        int sm_version = devProp.major * 10 + devProp.minor;
        String compileOptions[] = { "--gpu-architecture=compute_" + sm_version // probably not robust, but ok to
                                                                               // start.
        };
        for (String s : compileOptions) {
            System.out.println("  " + s);
        }
        if ((err = nvrtcCompileProgram(program, compileOptions.length, compileOptions)) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + errStr(err) + ") Unable to nvrtcCompileProgram");
            return true;
        }

        String programLog[] = new String[1];
        if ((err = nvrtcGetProgramLog(program, programLog)) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + errStr(err) + ") Unable to nvrtcGetProgramLog");
            return true;
        }
        if (!programLog[0].equals("")) {
            System.out.println("Program compilation log:\n" + programLog[0]);
        }

        String[] ptx = new String[1];

        if ((err = nvrtcGetPTX(program, ptx)) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + errStr(err) + ") Unable to nvrtcGetPTX");
            return true;
        }
        if ((err = nvrtcDestroyProgram(program)) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + errStr(err) + ") Unable to nvrtcDestroyProgram");
            return true;
        }
        System.out.println("Loading cuda kernels...");
        if ((err = JCudaDriver.cuModuleLoadData(module, ptx[0])) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + errStr(err) + ") failed to load ptx.");
            return true;
        }
        return false;
    }

    public static void render() {
        CUdeviceptr devPtr = mapResouce(sharedPbo.cudaPBOHandle());
        mandelbrot(devPtr, window.width(), window.height(), AppGL.textureWidth(), AppGL.textureHeight(), centerX,
                centerY, zoom, iterMult, doublePrecision);
        unmapResouce(sharedPbo.cudaPBOHandle());
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

    private static void mandelbrot(CUdeviceptr devPtr, int winWidth, int winHeight, int texWidth, int texHeight,
            double cx, double cy, double zoom, int iter, boolean doublePrec) {
        int blockSize = 16; // 256 threads per block
        int err;

        int mandelWidth = Math.min(winWidth, texWidth);
        int mandelHeight = Math.min(winHeight, texHeight);
        if (doublePrec) {
            Pointer doubleParams = Pointer.to(Pointer.to(devPtr), Pointer.to(new int[] { texWidth }),
                    Pointer.to(new int[] { texHeight }), Pointer.to(new int[] { mandelWidth }),
                    Pointer.to(new int[] { mandelHeight }), Pointer.to(new double[] { cx }),
                    Pointer.to(new double[] { cy }), Pointer.to(new double[] { zoom }), Pointer.to(new int[] { iter }));
            if ((err = JCudaDriver.cuLaunchKernel(mandelbrotDoubleKernel, texWidth / blockSize, texHeight / blockSize,
                    1, // grids
                    blockSize, blockSize, 1, // block
                    0, null, // shared memory, stream
                    doubleParams, null // params, extra
            )) != cudaError.cudaSuccess) {
                System.err.println("ERROR: (" + errStr(err) + ") in cuLaunchKernel for double_kernel");
            }
        } else {
            Pointer floatParams = Pointer.to(Pointer.to(devPtr), Pointer.to(new int[] { texWidth }),
                    Pointer.to(new int[] { texHeight }), Pointer.to(new int[] { mandelWidth }),
                    Pointer.to(new int[] { mandelHeight }), Pointer.to(new float[] { (float) cx }),
                    Pointer.to(new float[] { (float) cy }), Pointer.to(new float[] { (float) zoom }),
                    Pointer.to(new int[] { iter }));
            if ((err = JCudaDriver.cuLaunchKernel(mandelbrotFloatKernel, texWidth / blockSize, texHeight / blockSize, 1,
                    blockSize, blockSize, 1, 0, null, floatParams, null)) != cudaError.cudaSuccess) {
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
