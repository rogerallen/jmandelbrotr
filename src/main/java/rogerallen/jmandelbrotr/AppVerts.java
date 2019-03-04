package rogerallen.jmandelbrotr;

import static org.lwjgl.opengl.GL11.GL_TRIANGLE_STRIP;
import static org.lwjgl.opengl.GL11.glDrawArrays;
import static org.lwjgl.opengl.GL15.GL_ARRAY_BUFFER;
import static org.lwjgl.opengl.GL15.glBindBuffer;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL30.glGenVertexArrays;

public class AppVerts {
    private int id;
    
    private static AppVbo positionVbo, texCoordsVbo;
    
    private int numVerts;
    
    public AppVerts(int posAttr, float[] posCoords, int texAttr, float [] texCoords) {
        id = glGenVertexArrays();
        glBindVertexArray(id);
        positionVbo = new AppVbo(posAttr, posCoords);
        texCoordsVbo = new AppVbo(texAttr, texCoords);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        // FIXME: our VBOs are alls 2-component.
        numVerts = posCoords.length/2;
    }

    public void positionUpdate(float [] newCoords) {
        positionVbo.update(newCoords);
    }
    
    public void texCoordsUpdate(float [] newCoords) {
        texCoordsVbo.update(newCoords);
    }
    
    public void bind() {
        glBindVertexArray(id);
    }
    
    public void draw() {
        glDrawArrays(GL_TRIANGLE_STRIP, 0, numVerts);
    }
}
