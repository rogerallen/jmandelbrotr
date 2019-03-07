package rogerallen.jmandelbrotr;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;

public class AppMandelbrot {
    // realtime compile or load ptx?
    private final static boolean USE_REAL_TIME_COMPILE = true;

    private AppWindow window;
    private AppPbo sharedPbo;

    private double centerX, centerY, zoom;
    private int iterMult;
    private boolean doublePrecision;

    private CUfunction mandelbrotFloatKernel, mandelbrotDoubleKernel;

    public AppMandelbrot(AppWindow window, AppPbo sharedPbo) {
        this.window = window;
        this.sharedPbo = sharedPbo;
        centerX = -0.5;
        centerY = 0.0;
        zoom = 0.5;
        iterMult = 1;
        doublePrecision = false;
        mandelbrotFloatKernel = new CUfunction();
        mandelbrotDoubleKernel = new CUfunction();
    }

    public boolean init() {
        if (sharedPbo.registerBuffer()) {
            return true;
        }
        AppCUDAProgram prog = new AppCUDAProgram();
        if (USE_REAL_TIME_COMPILE) {
            if (prog.setupModule(App.RESOURCES_PREFIX + "mandelbrot.cu", true)) {
                return true;
            }
        } else {
            if (prog.setupModule("src/main/resources/mandelbrot.ptx", false)) {
                return true;
            }
        }
        if (prog.getFunction("mandel_float", mandelbrotFloatKernel)) {
            return true;
        }
        if (prog.getFunction("mandel_double", mandelbrotDoubleKernel)) {
            return true;
        }
        return false;
    }

    public void render() {
        CUdeviceptr devPtr = sharedPbo.mapGraphicsResource();
        mandelbrot(devPtr, window.width(), window.height(), AppGL.textureWidth(), AppGL.textureHeight(), centerX,
                centerY, zoom, iterMult, doublePrecision);
        sharedPbo.unmapGraphicsResource();
    }

    private void mandelbrot(CUdeviceptr devPtr, int winWidth, int winHeight, int texWidth, int texHeight, double cx,
            double cy, double zoom, int iter, boolean doublePrec) {
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
                System.err.println("ERROR: (" + AppCUDA.errStr(err) + ") in cuLaunchKernel for double_kernel");
            }
        } else {
            Pointer floatParams = Pointer.to(Pointer.to(devPtr), Pointer.to(new int[] { texWidth }),
                    Pointer.to(new int[] { texHeight }), Pointer.to(new int[] { mandelWidth }),
                    Pointer.to(new int[] { mandelHeight }), Pointer.to(new float[] { (float) cx }),
                    Pointer.to(new float[] { (float) cy }), Pointer.to(new float[] { (float) zoom }),
                    Pointer.to(new int[] { iter }));
            if ((err = JCudaDriver.cuLaunchKernel(mandelbrotFloatKernel, texWidth / blockSize, texHeight / blockSize, 1,
                    blockSize, blockSize, 1, 0, null, floatParams, null)) != cudaError.cudaSuccess) {
                System.err.println("ERROR: (" + AppCUDA.errStr(err) + ") in cuLaunchKernel for float_kernel");
            }
        }
        if ((err = JCuda.cudaGetLastError()) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + AppCUDA.errStr(err) + ") from cudaGetLastError");
        }
        if ((err = JCudaDriver.cuCtxSynchronize()) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + AppCUDA.errStr(err) + ") in cuCtxSynchronize");
        }
    }

    public void doublePrecision(boolean b) {
        doublePrecision = b;
    }

    public void iterMult(int i) {
        iterMult = i;
    }

    public void centerX(double d) {
        centerX = d;
    }

    public double centerX() {
        return centerX;
    }

    public void centerY(double d) {
        centerY = d;
    }

    public double centerY() {
        return centerY;
    }

    public double zoom() {
        return zoom;
    }

    public void zoomMul(double d) {
        zoom *= d;
    }

    public void zoomDiv(double d) {
        zoom /= d;
    }

}
