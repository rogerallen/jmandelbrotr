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

public class AppVbo {
    private int id;

    public AppVbo(int attr, float[] data) {
        // Note that this requires glBindVertexArray is active.
        id = glGenBuffers();
        FloatBuffer fb = (FloatBuffer) BufferUtils.createFloatBuffer(data.length).put(data).flip();
        glBindBuffer(GL_ARRAY_BUFFER, id);
        glBufferData(GL_ARRAY_BUFFER, fb, GL_DYNAMIC_DRAW);
        // NOTE: fixed to 2 float components
        glVertexAttribPointer(attr, 2, GL_FLOAT, false, 0, 0L);
        glEnableVertexAttribArray(attr);
    }

    public void update(float[] data) {
        FloatBuffer fb = (FloatBuffer) BufferUtils.createFloatBuffer(data.length).put(data).flip();
        glBindBuffer(GL_ARRAY_BUFFER, id);
        glBufferSubData(GL_ARRAY_BUFFER, 0, fb);
    }
}
