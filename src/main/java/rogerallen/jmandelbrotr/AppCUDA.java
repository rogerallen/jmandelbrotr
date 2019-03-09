package rogerallen.jmandelbrotr;

import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaGLDeviceList;

/**
 * AppCUDA class for some simple CUDA utility functions
 * 
 * @author rallen
 *
 */
public class AppCUDA {

    /**
     * set CUDA device to the first one. FIXME -- this should be more flexible.
     * 
     * @return true on error
     */
    public static boolean setDevice() {
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
        return false;
    }

    /**
     * Convert an err code intoa CUDA error name.
     * 
     * @param err
     * @return a helpful error string
     */
    public static String errStr(int err) {
        return JCuda.cudaGetErrorName(err) + "=" + err;
    }

}
