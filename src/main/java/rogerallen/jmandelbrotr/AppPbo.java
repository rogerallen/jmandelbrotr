package rogerallen.jmandelbrotr;

import static org.lwjgl.opengl.GL15.GL_DYNAMIC_COPY;
import static org.lwjgl.opengl.GL15.glBindBuffer;
import static org.lwjgl.opengl.GL15.glBufferData;
import static org.lwjgl.opengl.GL15.glGenBuffers;
import static org.lwjgl.opengl.GL21.GL_PIXEL_UNPACK_BUFFER;

import jcuda.runtime.cudaGraphicsResource;

public class AppPbo {
    private int id;
    private cudaGraphicsResource cudaPBOHandle;
    
    public AppPbo(int width, int height) {
        // Generate a buffer ID
        id = glGenBuffers();
        // Make this the current UNPACK buffer aka PBO (Pixel Buffer Object)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, id);
        // Allocate data for the buffer
        // TODO - try GL_DYNAMIC_DRAW
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, GL_DYNAMIC_COPY);

    }

    public int id() {
        return id;
    }

    public cudaGraphicsResource cudaPBOHandle() {
        return cudaPBOHandle;
    }
    public void cudaPBOHandle(cudaGraphicsResource cudaPBOHandle) {
        this.cudaPBOHandle = cudaPBOHandle;
    }

}
