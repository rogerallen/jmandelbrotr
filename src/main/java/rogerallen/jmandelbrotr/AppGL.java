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
import static org.lwjgl.opengl.GL11.GL_RGBA;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_2D;
import static org.lwjgl.opengl.GL11.GL_UNSIGNED_BYTE;
import static org.lwjgl.opengl.GL11.glBindTexture;
import static org.lwjgl.opengl.GL11.glClear;
import static org.lwjgl.opengl.GL11.glClearColor;
import static org.lwjgl.opengl.GL11.glReadPixels;
import static org.lwjgl.opengl.GL11.glTexSubImage2D;
import static org.lwjgl.opengl.GL11.glViewport;
import static org.lwjgl.opengl.GL12.GL_BGRA;

import java.io.IOException;
import java.nio.ByteBuffer;

import org.joml.Matrix4f;
import org.lwjgl.BufferUtils;
import org.lwjgl.opengl.GL;
import org.lwjgl.opengl.GLUtil;
import org.lwjgl.system.Callback;

public class AppGL {
    private static AppWindow window;
    private static AppVerts verts;
    private static AppProgram basicProg;
    private static AppTexture sharedTex;
    private static AppPbo sharedPbo;

    // private static GLCapabilities caps;
    private static Callback debugProc = null;

    private static Matrix4f cameraToView = new Matrix4f();

    public static boolean init(AppWindow appWindow, int maxWidth, int maxHeight, boolean debug) {
        window = appWindow;
        /* caps = */ GL.createCapabilities();
        if (debug) {
            debugProc = GLUtil.setupDebugMessageCallback();
        }
        glClearColor(1.0f, 1.0f, 0.5f, 0.0f);
        // Shared CUDA/GL pixel buffer
        sharedPbo = new AppPbo(maxWidth, maxHeight);
        // Create a GL Texture
        sharedTex = new AppTexture(maxWidth, maxHeight);
        try {
            basicProg = new AppProgram(App.RESOURCES_PREFIX + "basic_vert.glsl",
                    App.RESOURCES_PREFIX + "basic_frag.glsl");
        } catch (IOException e) {
            System.err.println("Error on shader setup: " + e);
            return true;
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
        float[] coords = { 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f };
        verts = new AppVerts(basicProg.attrPosition(), coords, basicProg.attrTexCoords(), coords);
        return false;
    }

    public static AppPbo sharedPbo() {
        return sharedPbo;
    }

    public static int textureWidth() {
        return sharedTex.width();
    }

    public static int textureHeight() {
        return sharedTex.height();
    }

    public static void handleResize() {
        glViewport(0, 0, window.width(), window.height());
        float wratio = (float) window.width() / sharedTex.width();
        float hratio = (float) window.height() / sharedTex.height();
        if (window.resized()) {
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

        // copy the CUDA-updated pixel buffer to the texture. Since the TexSubImage
        // pixels parameter (final one) is 0, Data is coming from a PBO, not host memory
        sharedPbo.bind();
        glBindTexture(GL_TEXTURE_2D, sharedTex.id());
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, sharedTex.width(), sharedTex.height(), GL_BGRA, GL_UNSIGNED_BYTE, 0);

        basicProg.bind();
        basicProg.updateCameraToView(cameraToView);
        verts.bind();
        verts.draw();

        verts.unbind();
        sharedPbo.unbind();
        basicProg.unbind();
    }

    public static ByteBuffer readPixels() {
        ByteBuffer buffer = BufferUtils.createByteBuffer(window.width() * window.height() * 4);
        glReadPixels(0, 0, window.width(), window.height(), GL_RGBA, GL_UNSIGNED_BYTE, buffer);
        return buffer;
    }

    public static void destroy() {
        if (debugProc != null) {
            debugProc.free();
        }
    }

}
