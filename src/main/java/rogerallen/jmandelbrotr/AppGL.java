package rogerallen.jmandelbrotr;

import static org.lwjgl.opengl.GL11.GL_COLOR_BUFFER_BIT;
import static org.lwjgl.opengl.GL11.GL_RGBA;
import static org.lwjgl.opengl.GL11.GL_UNSIGNED_BYTE;
import static org.lwjgl.opengl.GL11.glClear;
import static org.lwjgl.opengl.GL11.glClearColor;
import static org.lwjgl.opengl.GL11.glReadPixels;
import static org.lwjgl.opengl.GL11.glViewport;

import java.io.IOException;
import java.nio.ByteBuffer;

import org.joml.Matrix4f;
import org.lwjgl.BufferUtils;
import org.lwjgl.opengl.GL;
import org.lwjgl.opengl.GLUtil;
import org.lwjgl.system.Callback;

/**
 * OpenGL-related code for JMandelbrot.  This is a static class to reflect a single GL context.
 * 
 * - Creates a fullscreen quad (2 triangles)
 * - The quad has x,y and s,t coordinates & the upper-left corner is always 0,0 for both.
 * - When resized, 
 *   - the x,y for the larger axis ranges from 0-1 and the shorter axis 0-ratio where ratio is < 1.0
 *   - the s,t is a ratio of the window size to the shared CUDA/GL texture size.  
 *   - the shared CUDA/GL texture size should be set to the maximum size you expect. (Monitor width/height)
 * - These values are updated inside the vertex buffer.
 * 
 * @formatter:off
 * t y
 * 0 0 C--*--D triangle_strip ABCD
 *     |\....|
 *     |.\...|
 *     *..*..*
 *     |...\.|
 *     |....\|
 * 1 1 A--*--B
 *     0     1 x position coords
 *     0     1 s texture coords
 * @formatter:on
 *       
 * @author rallen
 *
 */
public class AppGL {
    private static AppWindow window;
    private static AppVerts verts;
    private static AppProgram basicProg;
    private static AppTexture sharedTex;
    private static AppPbo sharedPbo;

    // private static GLCapabilities caps;
    private static Callback debugProc = null;

    private static Matrix4f cameraToView = new Matrix4f();

    /**
     * initialize OpenGL, the shared pixel-buffer object (PBO) and the shared
     * texture & vertices.
     * 
     * @param appWindow is the AppWindow for this GL context.
     * @param maxWidth  is the size of the shared CUDA/GL buffer.
     * @param maxHeight is the size of the shared CUDA/GL buffer.
     * @param debug     is true if you desire GL debug messages from LWJGL
     * @return true on error
     */
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
        float[] coords = { 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f };
        verts = new AppVerts(basicProg.attrPosition(), coords, basicProg.attrTexCoords(), coords);
        return false;
    }

    /**
     * @return sharedPbo
     */
    public static AppPbo sharedPbo() {
        return sharedPbo;
    }

    /**
     * @return sharedTex width
     */
    public static int textureWidth() {
        return sharedTex.width();
    }

    /**
     * @return sharedTex height
     */
    public static int textureHeight() {
        return sharedTex.height();
    }

    /**
     * handle resize event & update the position & texcoord vertex buffers.
     */
    public static void handleResize() {
        glViewport(0, 0, window.width(), window.height());
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
            verts.updatePosition(new float[] { 0.0f, ypos, xpos, ypos, 0.0f, 0.0f, xpos, 0.0f });

            float wratio = (float) window.width() / sharedTex.width();
            float hratio = (float) window.height() / sharedTex.height();
            verts.updateTexCoords(new float[] { 0.0f, hratio, wratio, hratio, 0.0f, 0.0f, wratio, 0.0f });
        }

    }

    /**
     * copy the sharedPbo CUDA buffer to the sharedTex OpenGL texture & render it to
     * the screen.
     */
    public static void render() {
        glClear(GL_COLOR_BUFFER_BIT);

        // copy the CUDA-updated pixel buffer to the texture. 
        sharedPbo.bind();
        sharedTex.bind();
        sharedTex.copyFromPbo();

        basicProg.bind();
        basicProg.updateCameraToView(cameraToView);
        verts.bind();
        verts.draw();

        verts.unbind();
        sharedPbo.unbind();
        sharedTex.unbind();
        basicProg.unbind();
    }

    /**
     * read the rendered pixels
     * 
     * @return pixel RGBA unsigned byte ByteBuffer
     */
    public static ByteBuffer readPixels() {
        ByteBuffer buffer = BufferUtils.createByteBuffer(window.width() * window.height() * 4);
        glReadPixels(0, 0, window.width(), window.height(), GL_RGBA, GL_UNSIGNED_BYTE, buffer);
        return buffer;
    }

    /**
     * if we have a debugProc, free it.
     */
    public static void destroy() {
        if (debugProc != null) {
            debugProc.free();
        }
    }

}
