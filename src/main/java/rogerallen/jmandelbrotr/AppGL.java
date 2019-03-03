package rogerallen.jmandelbrotr;

/*
 * Some portions lifted from https://github.com/LWJGL/lwjgl3-demos/blob/master/src/org/lwjgl/demo/opengl/util/DemoUtils.java
 * and subject to this license.
 * 
 * Copyright © 2012-present Lightweight Java Game Library All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer. Redistributions in binary
 * form must reproduce the above copyright notice, this list of conditions and
 * the following disclaimer in the documentation and/or other materials provided
 * with the distribution. Neither the name Lightweight Java Game Library nor the
 * names of its contributors may be used to endorse or promote products derived
 * from this software without specific prior written permission. 
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
 * ANY EXPRESSOR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 */

import static org.lwjgl.opengl.GL11.GL_COLOR_BUFFER_BIT;
import static org.lwjgl.opengl.GL11.GL_FLOAT;
import static org.lwjgl.opengl.GL11.GL_LINEAR;
import static org.lwjgl.opengl.GL11.GL_RGBA;
import static org.lwjgl.opengl.GL11.GL_RGBA8;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_2D;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_MAG_FILTER;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_MIN_FILTER;
import static org.lwjgl.opengl.GL11.GL_TRIANGLE_STRIP;
import static org.lwjgl.opengl.GL11.GL_UNSIGNED_BYTE;
import static org.lwjgl.opengl.GL11.glBindTexture;
import static org.lwjgl.opengl.GL11.glClear;
import static org.lwjgl.opengl.GL11.glClearColor;
import static org.lwjgl.opengl.GL11.glDrawArrays;
import static org.lwjgl.opengl.GL11.glGenTextures;
import static org.lwjgl.opengl.GL11.glReadPixels;
import static org.lwjgl.opengl.GL11.glTexImage2D;
import static org.lwjgl.opengl.GL11.glTexParameteri;
import static org.lwjgl.opengl.GL11.glTexSubImage2D;
import static org.lwjgl.opengl.GL11.glViewport;
import static org.lwjgl.opengl.GL12.GL_BGRA;
import static org.lwjgl.opengl.GL15.GL_ARRAY_BUFFER;
import static org.lwjgl.opengl.GL15.GL_DYNAMIC_COPY;
import static org.lwjgl.opengl.GL15.GL_DYNAMIC_DRAW;
import static org.lwjgl.opengl.GL15.glBindBuffer;
import static org.lwjgl.opengl.GL15.glBufferData;
import static org.lwjgl.opengl.GL15.glGenBuffers;
import static org.lwjgl.opengl.GL15.glBufferSubData;
import static org.lwjgl.opengl.GL20.GL_COMPILE_STATUS;
import static org.lwjgl.opengl.GL20.GL_FRAGMENT_SHADER;
import static org.lwjgl.opengl.GL20.GL_LINK_STATUS;
import static org.lwjgl.opengl.GL20.GL_VERTEX_SHADER;
import static org.lwjgl.opengl.GL20.glAttachShader;
import static org.lwjgl.opengl.GL20.glCompileShader;
import static org.lwjgl.opengl.GL20.glCreateProgram;
import static org.lwjgl.opengl.GL20.glCreateShader;
import static org.lwjgl.opengl.GL20.glEnableVertexAttribArray;
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
import static org.lwjgl.opengl.GL20.glVertexAttribPointer;
import static org.lwjgl.opengl.GL21.GL_PIXEL_UNPACK_BUFFER;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL30.glGenVertexArrays;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.channels.FileChannel;

import org.joml.Matrix4f;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opengl.GL;
import org.lwjgl.opengl.GLUtil;
import org.lwjgl.system.Callback;

public class AppGL {

    public static String RESOURCES_PREFIX = "";

    // private static GLCapabilities caps;
    private static Callback debugProc;

    private static AppWindow window;

    private static int verts;
    private static int positionVbo, texCoordsVbo;
    private static int basicProg;
    private static int basicProgAttrPosition;
    private static int basicProgAttrTexCoords;
    private static int basicProgUniCameraToView;

    private static Matrix4f cameraToView = new Matrix4f();

    public static int sharedBufID, sharedTexID;
    public static int sharedTexWidth, sharedTexHeight;

    public static void init(AppWindow appWindow, int monitorWidth, int monitorHeight) throws IOException {
        window = appWindow;
        // FIXME -- I don't know how to configure Eclipse/Maven to do the right thing.
        // If I run in Eclipse, I load files as foo. If I run in a jar, I load files as
        // resources/foo
        // This is a hack workaround.
        InputStream source_in_jar = Thread.currentThread().getContextClassLoader()
                .getResourceAsStream("resources/mandelbrot.cu");
        if (source_in_jar != null) {
            RESOURCES_PREFIX = "resources/"; // JAR Compile
        }
        System.out.println("RESOURCES_PREFIX = \"" + RESOURCES_PREFIX + "\"");

        System.out.println("CUDA/GL Buffer size = " + monitorWidth + "x" + monitorHeight);

        /* caps = */ GL.createCapabilities();
        debugProc = GLUtil.setupDebugMessageCallback();
        glClearColor(1.0f, 1.0f, 0.5f, 0.0f);
        initTexture(monitorWidth, monitorHeight);
        initProgram();
        initVerts();
    }

    // Create a colored single fullscreen triangle
    // @formatter:off
	// t y
	// 0 0 C--*--D triangle_strip ABCD
	//     |\....|
	//     |.\...|
	//     *..*..*
	//     |...\.|
	//     |....\|
	// 1 1 A--*--B
	//     0     1 x position coords
	//     0     1 s texture coords
	// @formatter:on
    private static void initVerts() {
        verts = glGenVertexArrays();
        glBindVertexArray(verts);
        positionVbo = glGenBuffers();
        FloatBuffer fb = BufferUtils.createFloatBuffer(2 * 4);
        fb.put(0.0f).put(1.0f);
        fb.put(1.0f).put(1.0f);
        fb.put(0.0f).put(0.0f);
        fb.put(1.0f).put(0.0f);
        fb.flip();
        glBindBuffer(GL_ARRAY_BUFFER, positionVbo);
        glBufferData(GL_ARRAY_BUFFER, fb, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(basicProgAttrPosition, 2, GL_FLOAT, false, 0, 0L);
        glEnableVertexAttribArray(basicProgAttrPosition);
        texCoordsVbo = glGenBuffers();
        fb = BufferUtils.createFloatBuffer(2 * 4);
        fb.put(0.0f).put(1.0f);
        fb.put(1.0f).put(1.0f);
        fb.put(0.0f).put(0.0f);
        fb.put(1.0f).put(0.0f);
        fb.flip();
        glBindBuffer(GL_ARRAY_BUFFER, texCoordsVbo);
        glBufferData(GL_ARRAY_BUFFER, fb, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(basicProgAttrTexCoords, 2, GL_FLOAT, true, 0, 0L);
        glEnableVertexAttribArray(basicProgAttrTexCoords);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    private static ByteBuffer resizeBuffer(ByteBuffer buffer, int newCapacity) {
        ByteBuffer newBuffer = BufferUtils.createByteBuffer(newCapacity);
        buffer.flip();
        newBuffer.put(buffer);
        return newBuffer;
    }

    public static ByteBuffer ioResourceToByteBuffer(String resource) throws IOException {
        int bufferSize = 8192;
        ByteBuffer buffer;
        URL url = Thread.currentThread().getContextClassLoader().getResource(resource);
        File file = new File(url.getFile());
        if (file.isFile()) {
            FileInputStream fis = new FileInputStream(file);
            FileChannel fc = fis.getChannel();
            buffer = fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size());
            fc.close();
            fis.close();
        } else {
            buffer = BufferUtils.createByteBuffer(bufferSize);
            InputStream source = url.openStream();
            if (source == null)
                throw new FileNotFoundException(resource);
            try {
                byte[] buf = new byte[8192];
                while (true) {
                    int bytes = source.read(buf, 0, buf.length);
                    if (bytes == -1)
                        break;
                    if (buffer.remaining() < bytes)
                        buffer = resizeBuffer(buffer, buffer.capacity() * 2);
                    buffer.put(buf, 0, bytes);
                }
                buffer.flip();
            } finally {
                source.close();
            }
        }
        return buffer;
    }

    private static void initTexture(int monitorWidth, int monitorHeight) throws IOException {
        // Shared CUDA/GL texture
        // Shared OpenGL & CUDA buffer
        // Generate a buffer ID
        sharedBufID = glGenBuffers();
        sharedTexWidth = monitorWidth;
        sharedTexHeight = monitorHeight;
        // Make this the current UNPACK buffer aka PBO (Pixel Buffer Object)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, sharedBufID);
        // Allocate data for the buffer
        glBufferData(GL_PIXEL_UNPACK_BUFFER, sharedTexWidth * sharedTexHeight * 4, GL_DYNAMIC_COPY);

        // Create a GL Texture
        sharedTexID = glGenTextures();
        glBindTexture(GL_TEXTURE_2D, sharedTexID);
        // Allocate the texture memory.
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, sharedTexWidth, sharedTexHeight, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);
        // Set filter mode
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    private static int createShader(String resource, int type) throws IOException {
        int shader = glCreateShader(type);
        ByteBuffer source = ioResourceToByteBuffer(resource);
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

    private static void initProgram() throws IOException {
        int program = glCreateProgram();
        int vshader = createShader(RESOURCES_PREFIX + "basic_vert.glsl", GL_VERTEX_SHADER);
        int fshader = createShader(RESOURCES_PREFIX + "basic_frag.glsl", GL_FRAGMENT_SHADER);
        glAttachShader(program, vshader);
        glAttachShader(program, fshader);
        glLinkProgram(program);
        int linked = glGetProgrami(program, GL_LINK_STATUS);
        String programLog = glGetProgramInfoLog(program);
        if (programLog.trim().length() > 0)
            System.err.println(programLog);
        if (linked == 0)
            throw new AssertionError("Could not link program");
        glUseProgram(program);
        int texLocation = glGetUniformLocation(program, "texture");
        glUniform1i(texLocation, 0);
        basicProgAttrPosition = glGetAttribLocation(program, "position");
        basicProgAttrTexCoords = glGetAttribLocation(program, "texCoords");
        basicProgUniCameraToView = glGetUniformLocation(program, "cameraToView");
        glUseProgram(0);
        basicProg = program;
    }

    // Create a colored single fullscreen triangle
    // @formatter:off
	// t y
	// 0 0 C--*--D triangle_strip ABCD
	//     |\....|
	//     |.\...|
	//     *..*..*
	//     |...\.|
	//     |....\|
	// 1 1 A--*--B
	//     0     1 x position coords
	//     0     1 s texture coords
	// @formatter:on
    public static void handleResize() {
        glViewport(0, 0, window.width(), window.height());
        float winTexWidthRatio = (float) window.width() / sharedTexWidth;
        float winTexHeightRatio = (float) window.height() / sharedTexHeight;
        if (window.resized()) {
            // System.out.println("HANDLED "+window);
            window.resizeHandled();

            // anchor viewport to upper left corner (0, 0) to match the anchor on
            // the sharedTexture surface. See picture above.
            cameraToView.identity();
            float xpos = 1.0f, ypos = 1.0f;
            if (window.width() >= window.height()) {
                ypos = (float) window.height() / (float) window.width();
            } else {
                xpos = (float) window.width() / (float) window.height();
            }
            cameraToView.ortho(0.0f, xpos, ypos, 0.0f, -1, 1);

            // update on-screen triangles to reflect the aspect ratio change.
            FloatBuffer fb = BufferUtils.createFloatBuffer(2 * 4);
            fb.put(0.0f).put(ypos);
            fb.put(xpos).put(ypos);
            fb.put(0.0f).put(0.0f);
            fb.put(xpos).put(0.0f);
            fb.flip();
            glBindBuffer(GL_ARRAY_BUFFER, positionVbo);
            glBufferSubData(GL_ARRAY_BUFFER, 0, fb);
            fb = BufferUtils.createFloatBuffer(2 * 4);
            fb.put(0.0f * winTexWidthRatio).put(1.0f * winTexHeightRatio);
            fb.put(1.0f * winTexWidthRatio).put(1.0f * winTexHeightRatio);
            fb.put(0.0f * winTexWidthRatio).put(0.0f * winTexHeightRatio);
            fb.put(1.0f * winTexWidthRatio).put(0.0f * winTexHeightRatio);
            fb.flip();
            glBindBuffer(GL_ARRAY_BUFFER, texCoordsVbo);
            glBufferSubData(GL_ARRAY_BUFFER, 0, fb);
        }

    }

    public static void render() {
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(basicProg);
        FloatBuffer fb = BufferUtils.createFloatBuffer(16);
        cameraToView.get(fb);
        glUniformMatrix4fv(basicProgUniCameraToView, false, fb);
        glBindVertexArray(verts);

        // connect the pbo to the texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, sharedBufID);
        glBindTexture(GL_TEXTURE_2D, sharedTexID);
        // Since source parameter is NULL, Data is coming from a PBO, not host memory
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, sharedTexWidth, sharedTexHeight, GL_BGRA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glBindVertexArray(0);
        glUseProgram(0);
    }

    public static void destroy() {
        debugProc.free();
    }

    public static ByteBuffer getPixels() {
        ByteBuffer buffer = BufferUtils.createByteBuffer(window.width() * window.height() * 4);
        glReadPixels(0, 0, window.width(), window.height(), GL_RGBA, GL_UNSIGNED_BYTE, buffer);
        return buffer;
    }
}
