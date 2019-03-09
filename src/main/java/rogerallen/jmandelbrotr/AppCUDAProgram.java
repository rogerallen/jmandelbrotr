package rogerallen.jmandelbrotr;

import static jcuda.nvrtc.JNvrtc.nvrtcCompileProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcCreateProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcDestroyProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcGetPTX;
import static jcuda.nvrtc.JNvrtc.nvrtcGetProgramLog;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.nvrtc.nvrtcProgram;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.cudaError;

/**
 * Methods related to compiling .cu files or loading .ptx files & accessing
 * routines within them.
 * 
 * @author rallen
 *
 */
public class AppCUDAProgram {
    CUmodule module;

    /**
     * constructor just nulls module.
     */
    public AppCUDAProgram() {
        module = null;
    }

    /**
     * Compile .cu file or load ptx file and update module.
     * 
     * @param filePath - .cu or .ptx file path.
     * @param compile  - compile .cu (true) or load .ptx (false)
     * @return true if error
     */
    public boolean setupModule(String filePath, boolean compile) {
        assert (module == null);
        module = new CUmodule();
        if (compile) {
            if (compileCuda(filePath)) {
                return true;
            }
        } else {
            if (loadPtx(filePath)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Take in aFunction name and store in aKernel. AKernel must be a reference to a
     * previously-constructed CUfunction.
     * 
     * @param aFunction - function name inside the module
     * @param aKernel   - on exit, will contain reference to aFunction.
     * @return true if error
     */
    public boolean getFunction(String aFunction, CUfunction aKernel) {
        assert (module != null);
        assert (aKernel != null);
        int err;
        if ((err = JCudaDriver.cuModuleGetFunction(aKernel, module, aFunction)) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + AppCUDA.errStr(err) + ") failed to get function " + aFunction);
            return true;
        }
        return false;
    }

    /**
     * load a ptx file into module.
     * 
     * @param filename is the path to the ptx file.
     * @return true if error
     */
    private boolean loadPtx(String filename) {
        int err;
        System.out.println("Loading ptx directly...");
        if ((err = JCudaDriver.cuModuleLoad(module, filename)) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + AppCUDA.errStr(err) + ") failed to find " + filename);
            return true;
        }
        return false;
    }

    /**
     * compile a cuda file & load into module.
     * 
     * @param filename is the path to the cuda file.
     * @return true if error
     */
    private boolean compileCuda(String filename) {
        int err;
        System.out.println("Compiling cuda kernels...");
        String programSourceCode;
        try {
            programSourceCode = StandardCharsets.UTF_8.decode(AppUtils.ioResourceToByteBuffer(filename)).toString();
        } catch (IOException e) {
            System.err.println("Error loading " + filename + " " + e);
            return true;
        }
        // Use the NVRTC to create a program by compiling the source code
        nvrtcProgram program = new nvrtcProgram();
        if ((err = nvrtcCreateProgram(program, programSourceCode, null, 0, null, null)) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + AppCUDA.errStr(err) + ") Unable to nvrtcCreateProgram");
            return true;
        }
        cudaDeviceProp devProp = new cudaDeviceProp();
        if ((err = JCuda.cudaGetDeviceProperties(devProp, 0)) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + AppCUDA.errStr(err) + ") Unable to cudaGetDeviceProperties");
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
            System.err.println("ERROR: (" + AppCUDA.errStr(err) + ") Unable to nvrtcCompileProgram");
            return true;
        }

        String programLog[] = new String[1];
        if ((err = nvrtcGetProgramLog(program, programLog)) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + AppCUDA.errStr(err) + ") Unable to nvrtcGetProgramLog");
            return true;
        }
        if (!programLog[0].equals("")) {
            System.out.println("Program compilation log:\n" + programLog[0]);
        }

        String[] ptx = new String[1];

        if ((err = nvrtcGetPTX(program, ptx)) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + AppCUDA.errStr(err) + ") Unable to nvrtcGetPTX");
            return true;
        }
        if ((err = nvrtcDestroyProgram(program)) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + AppCUDA.errStr(err) + ") Unable to nvrtcDestroyProgram");
            return true;
        }
        System.out.println("Loading cuda kernels...");
        if ((err = JCudaDriver.cuModuleLoadData(module, ptx[0])) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + AppCUDA.errStr(err) + ") failed to load ptx.");
            return true;
        }
        return false;
    }

}
