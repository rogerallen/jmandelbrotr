package rogerallen.jmandelbrotr;

import static org.lwjgl.opengl.GL15.GL_DYNAMIC_DRAW;
import static org.lwjgl.opengl.GL15.glBindBuffer;
import static org.lwjgl.opengl.GL15.glBufferData;
import static org.lwjgl.opengl.GL15.glGenBuffers;
import static org.lwjgl.opengl.GL21.GL_PIXEL_UNPACK_BUFFER;

import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUgraphicsResource;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaGraphicsRegisterFlags;
import jcuda.runtime.cudaGraphicsResource;

/**
 * Pixel-buffer Object (PBO) Handling. CUDA writes to the buffer, OpenGL reads
 * from it. When OpenGL reads it, it expects it to be a BGRA unsigned byte
 * format buffer.
 * 
 * @author rallen
 *
 */
public class AppPbo {
    private int id;
    private cudaGraphicsResource cudaPBOHandle;

    /**
     * create RGBA unsigned byte pixel buffer object
     * 
     * @param width
     * @param height
     */
    public AppPbo(int width, int height) {
        // Generate a buffer ID
        id = glGenBuffers();
        cudaPBOHandle = null;
        // Make this the current UNPACK buffer aka PBO (Pixel Buffer Object)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, id);
        // Allocate data for the buffer. DYNAMIC (modified repeatedly) DRAW (not reading
        // from GL)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, GL_DYNAMIC_DRAW);

    }

    /**
     * register this buffer so CUDA can use it. This happens once at the start.
     * 
     * @return true on error
     */
    public boolean registerBuffer() {
        int err;
        // CUDA writes to the buffer, OpenGL reads, then this repeats.
        // So, add WriteDiscard flag to this buffer.
        cudaGraphicsResource cudaPBOHandle = new cudaGraphicsResource();
        if ((err = JCuda.cudaGraphicsGLRegisterBuffer(cudaPBOHandle, this.id,
                cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard)) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + AppCUDA.errStr(err) + ") Failed to register buffer " + this.id);
            System.err.println("Make sure that you are running graphics on NVIDIA GPU");
            return true;
        }
        this.cudaPBOHandle = cudaPBOHandle;
        return false;
    }

    /**
     * map the PBO as a graphics resource so CUDA can write into it.
     * 
     * @return pointer to the PBO on the device.
     */
    public CUdeviceptr mapGraphicsResource() {
        assert (cudaPBOHandle != null);
        // TODO - consider moving these externally? Should we be doing new at 60Hz?
        CUgraphicsResource cuResource = new CUgraphicsResource(cudaPBOHandle);
        CUdeviceptr basePointer = new CUdeviceptr();
        int err;
        if ((err = JCudaDriver.cuGraphicsMapResources(1, new CUgraphicsResource[] { cuResource },
                null)) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + AppCUDA.errStr(err) + ") failed to map resource");
        }
        if ((err = JCudaDriver.cuGraphicsResourceGetMappedPointer(basePointer, new long[1],
                cuResource)) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + AppCUDA.errStr(err) + ") failed to get mapped pointer");
        }
        return basePointer;
    }

    /**
     * unmap the PBO so that OpenGL can read from it.
     */
    // FIXME -- add error handling
    public void unmapGraphicsResource() {
        assert (cudaPBOHandle != null);
        CUgraphicsResource cuResource = new CUgraphicsResource(cudaPBOHandle);
        int err;
        if ((err = JCudaDriver.cuGraphicsUnmapResources(1, new CUgraphicsResource[] { cuResource },
                null)) != cudaError.cudaSuccess) {
            System.err.println("ERROR: (" + AppCUDA.errStr(err) + ") failed to unmap resource");
        }
    }

    /**
     * bind the PBO for OpenGL's use.
     */
    public void bind() {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, this.id);
    }

    /**
     * unbind the PBO so OpenGL does not use it.
     */
    public void unbind() {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }

}
