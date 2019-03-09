package rogerallen.jmandelbrotr;

import static org.lwjgl.opengl.GL11.GL_LINEAR;
import static org.lwjgl.opengl.GL11.GL_RGBA8;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_2D;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_MAG_FILTER;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_MIN_FILTER;
import static org.lwjgl.opengl.GL11.GL_UNSIGNED_BYTE;
import static org.lwjgl.opengl.GL11.glBindTexture;
import static org.lwjgl.opengl.GL11.glGenTextures;
import static org.lwjgl.opengl.GL11.glTexImage2D;
import static org.lwjgl.opengl.GL11.glTexParameteri;
import static org.lwjgl.opengl.GL11.glTexSubImage2D;
import static org.lwjgl.opengl.GL12.GL_BGRA;

/**
 * OpenGL Texture Container class. Fixed to BGRA unsigned byte format.
 * 
 * @author rallen
 *
 */
public class AppTexture {
    private int id;
    private int width, height;

    /**
     * construct BGRA unsigned byte texture
     * 
     * @param width
     * @param height
     */
    public AppTexture(int width, int height) {
        id = glGenTextures();
        this.width = width;
        this.height = height;
        glBindTexture(GL_TEXTURE_2D, id);
        // Allocate the texture memory. This will be filled in by the PBO during
        // rendering
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);
        // Set filter mode
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    /**
     * copy from the pixel buffer object to this texture. Since the TexSubImage
     * pixels parameter (final one) is 0, Data is coming from a PBO, not host memory
     */
    public void copyFromPbo() {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, 0);
    }

    /**
     * @return texture width
     */
    public int width() {
        return width;
    }

    /**
     * @return texture height
     */
    public int height() {
        return height;
    }

    /**
     * bind this texture so OpenGL will use it
     */
    public void bind() {
        glBindTexture(GL_TEXTURE_2D, id);
    }

    /**
     * unbind this texture so OpenGL will stop using it
     */
    public void unbind() {
        glBindTexture(GL_TEXTURE_2D, 0);
    }

}
