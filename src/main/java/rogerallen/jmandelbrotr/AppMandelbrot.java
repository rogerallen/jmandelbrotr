package rogerallen.jmandelbrotr;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;

/**
 * Code for rendering a Mandelbrot function via CUDA into a sharedPbo.
 * 
 * @author rallen
 *
 */
public class AppMandelbrot {
    // realtime compile or load ptx?
    private final static boolean USE_REAL_TIME_COMPILE = true;

    private AppWindow window;
    private AppPbo sharedPbo;

    // controls for the renderer
    private double centerX, centerY, zoom;
    private int iterMult;
    private boolean doublePrecision;

    // we can run a single or double-precision mandelbrot kernel.
    private CUfunction mandelbrotFloatKernel, mandelbrotDoubleKernel;

    /**
     * Constructor that initializes the renderer to default values.
     * 
     * @param window    the window associated with the render (so we can find the
     *                  current window width, height)
     * @param sharedPbo the pixel-buffer object to render into
     */
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

    /**
     * Register our PBO with CUDA, compile or load our code & setup our kernels.
     * 
     * @return true if error
     */
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

    /**
     * map the sharedPbo, render the mandelbrot into it & then unmap it so OpenGL
     * can use it.
     */
    public void render() {
        CUdeviceptr devPtr = sharedPbo.mapGraphicsResource();
        mandelbrot(devPtr, window.width(), window.height(), AppGL.textureWidth(), AppGL.textureHeight());
        sharedPbo.unmapGraphicsResource();
    }

    /**
     * run the CUDA mandelbrot kernel to store the current view into devPtr
     * 
     * @param devPtr    points to the sharedPbo on the device.
     * @param winWidth  is the current window width (always < texWidth)
     * @param winHeight is the current window height (always < texHeight)
     * @param texWidth  is the sharedPbo width
     * @param texHeight is the sharedPbo height
     */
    private void mandelbrot(CUdeviceptr devPtr, int winWidth, int winHeight, int texWidth, int texHeight) {
        int blockSize = 16; // 256 threads per block
        int err;

        int mandelWidth = Math.min(winWidth, texWidth);
        int mandelHeight = Math.min(winHeight, texHeight);
        if (doublePrecision) {
            Pointer doubleParams = Pointer.to(Pointer.to(devPtr), Pointer.to(new int[] { texWidth }),
                    Pointer.to(new int[] { texHeight }), Pointer.to(new int[] { mandelWidth }),
                    Pointer.to(new int[] { mandelHeight }), Pointer.to(new double[] { centerX }),
                    Pointer.to(new double[] { centerY }), Pointer.to(new double[] { zoom }),
                    Pointer.to(new int[] { iterMult }));
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
                    Pointer.to(new int[] { mandelHeight }), Pointer.to(new float[] { (float) centerX }),
                    Pointer.to(new float[] { (float) centerY }), Pointer.to(new float[] { (float) zoom }),
                    Pointer.to(new int[] { iterMult }));
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

    /**
     * set doublePrecision kernel mode
     * 
     * @param b new doublePrecision mode
     */
    public void doublePrecision(boolean b) {
        doublePrecision = b;
    }

    /**
     * set iterMult - a multiplier on the iteration count to get i*256 levels of
     * colors
     * 
     * @param i - new value for iteration multiplier
     */
    public void iterMult(int i) {
        iterMult = i;
    }

    /**
     * set the center X coord of the Mandelbrot set window
     * 
     * @param d new value
     */
    public void centerX(double d) {
        centerX = d;
    }

    /**
     * @return center X coord of the Mandelbrot set window
     */
    public double centerX() {
        return centerX;
    }

    /**
     * set the center Y coord of the Mandelbrot set window
     * 
     * @param d new value
     */
    public void centerY(double d) {
        centerY = d;
    }

    /**
     * @return center Y coord of the Mandelbrot set window
     */
    public double centerY() {
        return centerY;
    }

    /**
     * @return current Mandelbrot zoom level
     */
    public double zoom() {
        return zoom;
    }

    /**
     * adjust the mandelbrot zoom level
     * 
     * @param d is the multiplier
     */
    public void zoomMul(double d) {
        zoom *= d;
    }

    /**
     * adjust the mandelbrot zoom level
     * 
     * @param d is the divisor
     */
    public void zoomDiv(double d) {
        zoom /= d;
    }

}
