package rogerallen.jmandelbrotr;

import static org.lwjgl.opengl.GL11.GL_TRIANGLE_STRIP;
import static org.lwjgl.opengl.GL11.glDrawArrays;
import static org.lwjgl.opengl.GL15.GL_ARRAY_BUFFER;
import static org.lwjgl.opengl.GL15.glBindBuffer;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL30.glGenVertexArrays;

/**
 * Vertex Array container class that contains our VertexBufferObjects.
 * Hard-coded to have only position & texCoord VBOs and must be arranged as a
 * triangle strip.
 * 
 * @author rallen
 *
 */
public class AppVerts {
    private int id;

    private static AppVbo positionVbo, texCoordsVbo;

    private int numVerts;

    /**
     * Constructor
     * 
     * @param posAttr   is the vertex shader attribute index for position
     * @param posCoords is the float array for X,Y positions
     * @param texAttr   is the vertex shader attribute index for texCoords
     * @param texCoords is the float array for S,T texCoords
     */
    public AppVerts(int posAttr, float[] posCoords, int texAttr, float[] texCoords) {
        id = glGenVertexArrays();
        glBindVertexArray(id);
        positionVbo = new AppVbo(posAttr, posCoords);
        texCoordsVbo = new AppVbo(texAttr, texCoords);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        // NOTE our VBOs are all 2-component (X,Y or S,T).
        numVerts = posCoords.length / 2;
    }

    /**
     * draw our Vertex Array as a Triangle Strip
     */
    public void draw() {
        glDrawArrays(GL_TRIANGLE_STRIP, 0, numVerts);
    }

    /**
     * Update the position VBO. Must be the same size as the original VBO
     * 
     * @param newCoords is the float array for X,Y positions
     */
    public void updatePosition(float[] newCoords) {
        assert (2 * numVerts == newCoords.length);
        positionVbo.update(newCoords);
    }

    /**
     * Update the texCoord VBO. Must be the same size as the original VBO
     * 
     * @param newCoords is the float array for S,T texCoords
     */
    public void updateTexCoords(float[] newCoords) {
        assert (2 * numVerts == newCoords.length);
        texCoordsVbo.update(newCoords);
    }

    /**
     * bind this VertexArray so OpenGL can use it
     */
    public void bind() {
        glBindVertexArray(id);
    }

    /**
     * unbind this VertexArray so OpenGL will stop using it
     */
    public void unbind() {
        glBindVertexArray(0);
    }

}
