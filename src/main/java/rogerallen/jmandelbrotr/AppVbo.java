package rogerallen.jmandelbrotr;

import static org.lwjgl.opengl.GL11.GL_FLOAT;
import static org.lwjgl.opengl.GL15.GL_ARRAY_BUFFER;
import static org.lwjgl.opengl.GL15.GL_DYNAMIC_DRAW;
import static org.lwjgl.opengl.GL15.glBindBuffer;
import static org.lwjgl.opengl.GL15.glBufferData;
import static org.lwjgl.opengl.GL15.glBufferSubData;
import static org.lwjgl.opengl.GL15.glGenBuffers;
import static org.lwjgl.opengl.GL20.glEnableVertexAttribArray;
import static org.lwjgl.opengl.GL20.glVertexAttribPointer;

import java.nio.FloatBuffer;

import org.lwjgl.BufferUtils;

/**
 * Vertex Buffer Object (VBO) container class.
 * 
 * @author rallen
 *
 */
public class AppVbo {
    private int id;
    private int originalLength;

    /**
     * Constructor for VBO (requires that glBindVertexArray is active)
     * 
     * @param attr is the index of this data for the Vertex Shader
     * @param data is the float array of incoming data.
     */
    public AppVbo(int attr, float[] data) {
        // Note that this requires glBindVertexArray is active.
        id = glGenBuffers();
        originalLength = data.length;
        FloatBuffer fb = (FloatBuffer) BufferUtils.createFloatBuffer(data.length).put(data).flip();
        glBindBuffer(GL_ARRAY_BUFFER, id);
        glBufferData(GL_ARRAY_BUFFER, fb, GL_DYNAMIC_DRAW);
        // NOTE: fixed to 2 float components
        glVertexAttribPointer(attr, 2, GL_FLOAT, false, 0, 0L);
        glEnableVertexAttribArray(attr);
    }

    /**
     * Overwrite the VBO with new data.  Size must be the same as what was constructed.
     * @param data the float array of new data.
     */
    public void update(float[] data) {
        assert(data.length == originalLength);
        FloatBuffer fb = (FloatBuffer) BufferUtils.createFloatBuffer(data.length).put(data).flip();
        glBindBuffer(GL_ARRAY_BUFFER, id);
        glBufferSubData(GL_ARRAY_BUFFER, 0, fb);
    }
}
