package rogerallen.jmandelbrotr;

import static org.lwjgl.opengl.GL20.GL_COMPILE_STATUS;
import static org.lwjgl.opengl.GL20.GL_FRAGMENT_SHADER;
import static org.lwjgl.opengl.GL20.GL_LINK_STATUS;
import static org.lwjgl.opengl.GL20.GL_VERTEX_SHADER;
import static org.lwjgl.opengl.GL20.glAttachShader;
import static org.lwjgl.opengl.GL20.glCompileShader;
import static org.lwjgl.opengl.GL20.glCreateProgram;
import static org.lwjgl.opengl.GL20.glCreateShader;
import static org.lwjgl.opengl.GL20.glGetAttribLocation;
import static org.lwjgl.opengl.GL20.glGetProgramInfoLog;
import static org.lwjgl.opengl.GL20.glGetProgrami;
import static org.lwjgl.opengl.GL20.glGetShaderInfoLog;
import static org.lwjgl.opengl.GL20.glGetShaderi;
import static org.lwjgl.opengl.GL20.glGetUniformLocation;
import static org.lwjgl.opengl.GL20.glLinkProgram;
import static org.lwjgl.opengl.GL20.glShaderSource;
import static org.lwjgl.opengl.GL20.glUniform1i;
import static org.lwjgl.opengl.GL20.glUniformMatrix4fv;
import static org.lwjgl.opengl.GL20.glUseProgram;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import org.joml.Matrix4f;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;

/**
 * OpenGL Shader Program handling code.
 * 
 * This shader is fixed to be a simple 2D position & texcoord vertex shader &
 * single texture-loading fragment shader.
 * 
 * @author rallen
 *
 */
public class AppProgram {
    private int id;
    private int attrPosition;
    private int attrTexCoords;
    private int uniCameraToView;

    /**
     * Load, compile & setup the attributes & uniform value for this shader.
     * 
     * @param vertProgramPath
     * @param fragProgramPath
     * @throws IOException
     */
    // FIXME constructor should not throw exception.
    public AppProgram(String vertProgramPath, String fragProgramPath) throws IOException {
        id = glCreateProgram();
        int vshader = createShader(vertProgramPath, GL_VERTEX_SHADER);
        int fshader = createShader(fragProgramPath, GL_FRAGMENT_SHADER);
        glAttachShader(id, vshader);
        glAttachShader(id, fshader);
        glLinkProgram(id);
        int linked = glGetProgrami(id, GL_LINK_STATUS);
        String programLog = glGetProgramInfoLog(id);
        if (programLog.trim().length() > 0)
            System.err.println(programLog);
        if (linked == 0)
            throw new AssertionError("Could not link program");
        glUseProgram(id);
        int texLocation = glGetUniformLocation(id, "texture");
        glUniform1i(texLocation, 0);
        attrPosition = glGetAttribLocation(id, "position");
        attrTexCoords = glGetAttribLocation(id, "texCoords");
        uniCameraToView = glGetUniformLocation(id, "cameraToView");
        glUseProgram(0);
    }

    /**
     * Load & compile a single shader file
     * 
     * @param resource path to the resource
     * @param type     type of shader (VS, or FS)
     * @return shader ID
     * @throws IOException
     */
    private int createShader(String resource, int type) throws IOException {
        int shader = glCreateShader(type);
        ByteBuffer source = AppUtils.ioResourceToByteBuffer(resource);
        PointerBuffer strings = BufferUtils.createPointerBuffer(1);
        IntBuffer lengths = BufferUtils.createIntBuffer(1);
        strings.put(0, source);
        lengths.put(0, source.remaining());
        glShaderSource(shader, strings, lengths);
        glCompileShader(shader);
        int compiled = glGetShaderi(shader, GL_COMPILE_STATUS);
        String shaderLog = glGetShaderInfoLog(shader);
        if (shaderLog.trim().length() > 0) {
            System.err.println(shaderLog);
        }
        if (compiled == 0) {
            throw new AssertionError("Could not compile shader");
        }
        return shader;
    }

    /**
     * bind this program so OpenGL will use it.
     */
    public void bind() {
        glUseProgram(id);
    }

    /**
     * unbind this program so OpenGL will not use it.
     */
    public void unbind() {
        glUseProgram(0);
    }

    /**
     * update the camera-to-view matrix uniform.
     * 
     * @param cameraToView
     */
    public void updateCameraToView(Matrix4f cameraToView) {
        FloatBuffer fb = BufferUtils.createFloatBuffer(16);
        cameraToView.get(fb);
        glUniformMatrix4fv(uniCameraToView, false, fb);
    }

    /**
     * @return attribute position ID for the Vertex Buffer Object
     */
    public int attrPosition() {
        return attrPosition;
    }

    /**
     * @return attribute texCoord ID for the Vertex Buffer Object
     */
    public int attrTexCoords() {
        return attrTexCoords;
    }

}
