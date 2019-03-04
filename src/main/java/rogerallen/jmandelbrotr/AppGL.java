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

    // private static GLCapabilities caps;
    private static Callback debugProc;

    private static AppWindow window;
    private static AppVerts verts;
    private static AppProgram basicProg;
    private static AppTexture sharedTex;

    private static Matrix4f cameraToView = new Matrix4f();
    
    public static int sharedBufID;

    public static void init(AppWindow appWindow, int maxWidth, int maxHeight) throws IOException {

        System.out.println("CUDA/GL Buffer size = " + maxWidth + "x" + maxHeight);

        window = appWindow;
        /* caps = */ GL.createCapabilities();
        debugProc = GLUtil.setupDebugMessageCallback();
        glClearColor(1.0f, 1.0f, 0.5f, 0.0f);
        initTexture(maxWidth, maxHeight);
        basicProg = new AppProgram(
                App.RESOURCES_PREFIX + "basic_vert.glsl",
                App.RESOURCES_PREFIX + "basic_frag.glsl");
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
        float[] coords = { 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f };
        verts = new AppVerts(
                basicProg.attrPosition(), coords, 
                basicProg.attrTexCoords(), coords);
    }

    // FIXME move to buffer utils class
    private static ByteBuffer resizeBuffer(ByteBuffer buffer, int newCapacity) {
        ByteBuffer newBuffer = BufferUtils.createByteBuffer(newCapacity);
        buffer.flip();
        newBuffer.put(buffer);
        return newBuffer;
    }
    
    // FIXME move to buffer utils class
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

    private static void initTexture(int maxWidth, int maxHeight) throws IOException {
        // Shared CUDA/GL texture
        // Shared OpenGL & CUDA buffer
        // Generate a buffer ID
        sharedBufID = glGenBuffers();
        // Make this the current UNPACK buffer aka PBO (Pixel Buffer Object)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, sharedBufID);
        // Allocate data for the buffer
        // FIXME - try GL_DYNAMIC_DRAW
        glBufferData(GL_PIXEL_UNPACK_BUFFER, maxWidth * maxHeight * 4, GL_DYNAMIC_COPY);

        // Create a GL Texture
        sharedTex = new AppTexture(maxWidth, maxHeight);
    }
    
    public static int textureWidth() {
        return sharedTex.width();
    }
    public static int textureHeight() {
        return sharedTex.height();
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
        float wratio = (float) window.width() / sharedTex.width();
        float hratio = (float) window.height() / sharedTex.height();
        if (window.resized()) {
            // System.out.println("HANDLED "+window);
            window.resizeHandled();

            // anchor viewport to upper left corner (0, 0) to match the anchor on
            // the sharedTexture surface. See picture above.
            float xpos = 1.0f, ypos = 1.0f;
            if (window.width() >= window.height()) {
                ypos = (float) window.height() / (float) window.width();
            } else {
                xpos = (float) window.width() / (float) window.height();
            }
            cameraToView.identity();
            cameraToView.ortho(0.0f, xpos, ypos, 0.0f, -1, 1);

            // update on-screen triangles to reflect the aspect ratio change.
            verts.positionUpdate(new float[] { 0.0f, ypos, xpos, ypos, 0.0f, 0.0f, xpos, 0.0f });
            verts.texCoordsUpdate(new float[] { 0.0f, hratio, wratio, hratio, 0.0f, 0.0f, wratio, 0.0f });
        }

    }

    public static void render() {
        glClear(GL_COLOR_BUFFER_BIT);

        basicProg.use();
        basicProg.updateCameraToView(cameraToView);
        
        verts.bind();

        // connect the pbo to the texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, sharedBufID);
        glBindTexture(GL_TEXTURE_2D, sharedTex.id());
        // Since the pixels parameter (final one) is 0, Data is coming from a PBO, not host memory
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, sharedTex.width(), sharedTex.height(), GL_BGRA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        verts.draw();

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
