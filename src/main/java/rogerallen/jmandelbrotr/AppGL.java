package rogerallen.jmandelbrotr;

import static org.lwjgl.opengl.GL11.*;
import static org.lwjgl.opengl.GL15.*;
import static org.lwjgl.opengl.GL20.*;
import static org.lwjgl.opengl.GL30.*;
import static org.lwjgl.stb.STBImage.stbi_image_free;
import static org.lwjgl.stb.STBImage.stbi_load_from_memory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
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

	public static int window_width = 800;
	public static int window_height = 800;
	public static boolean window_resized = false;

	private static int verts;
	private static int basic_prog;
	private static int basic_prog_a_position;
	private static int basic_prog_a_texCoords;
	private static int basic_prog_u_cameraToView;

	private static Matrix4f cameraToView = new Matrix4f();
	
	public static final int SHARED_TEX_SIZE = 2048;
	public static int shared_buf_id, shared_tex_id;
	public static int shared_tex_width;
	public static int shared_tex_height;

	public static void init() throws IOException {
		/* caps = */ GL.createCapabilities();
		debugProc = GLUtil.setupDebugMessageCallback();
		glClearColor(1.0f, 1.0f, 0.5f, 0.0f);
		initTexture();
		initProgram();
		initVerts();
	}

	// Create a colored single fullscreen triangle
	// @formatter:off
	// 3  *______________
	//    |\_____________
	//    | \____________
	// 2  *  \___________
	//    |   \__________
	//    |    \_________
	// 1  *--*--*________
	//    |.....|\_______
	//    |.....| \______
	// 0  *..*..*  \_____
	//    |.....|   \____
	//    |.....|    \___
	// -1 *--*--*--*--*__
	//   -1  0  1  2  3 x position coords
	// 0 1 2 s texture coords
	// @formatter:on
	private static void initVerts() {
		verts = glGenVertexArrays();
		glBindVertexArray(verts);
		int positionVbo = glGenBuffers();
		FloatBuffer fb = BufferUtils.createFloatBuffer(2 * 3);
		fb.put(-1.0f).put(3.0f);
		fb.put(-1.0f).put(-1.0f);
		fb.put(3.0f).put(-1.0f);
		fb.flip();
		glBindBuffer(GL_ARRAY_BUFFER, positionVbo);
		glBufferData(GL_ARRAY_BUFFER, fb, GL_STATIC_DRAW);
		glVertexAttribPointer(basic_prog_a_position, 2, GL_FLOAT, false, 0, 0L);
		glEnableVertexAttribArray(basic_prog_a_position);
		int texCoordsVbo = glGenBuffers();
		fb = BufferUtils.createFloatBuffer(2 * 3);
		fb.put(0.0f).put(-1.0f);
		fb.put(0.0f).put(1.0f);
		fb.put(2.0f).put(1.0f);
		fb.flip();
		glBindBuffer(GL_ARRAY_BUFFER, texCoordsVbo);
		glBufferData(GL_ARRAY_BUFFER, fb, GL_STATIC_DRAW);
		glVertexAttribPointer(basic_prog_a_texCoords, 2, GL_FLOAT, true, 0, 0L);
		glEnableVertexAttribArray(basic_prog_a_texCoords);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

	private static ByteBuffer resourceToByteBuffer(String resource) throws IOException {
		URL url = Thread.currentThread().getContextClassLoader().getResource(resource);
		File file = new File(url.getFile());
		FileInputStream fis = new FileInputStream(file);
		FileChannel fc = fis.getChannel();
		ByteBuffer buffer = fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size());
		fc.close();
		fis.close();
		return buffer;
	}

	private static void initTexture() throws IOException {
		// Original texture (remove?)
		IntBuffer width = BufferUtils.createIntBuffer(1);
		IntBuffer height = BufferUtils.createIntBuffer(1);
		IntBuffer components = BufferUtils.createIntBuffer(1);
		ByteBuffer data = stbi_load_from_memory(resourceToByteBuffer("side1.png"), width, height, components, 4);
		int id = glGenTextures();
		glBindTexture(GL_TEXTURE_2D, id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width.get(), height.get(), 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
		stbi_image_free(data);
		
		// Shared CUDA/GL texture
        // Shared OpenGL & CUDA buffer
        // Generate a buffer ID
		shared_buf_id = glGenBuffers();
		shared_tex_width = shared_tex_height = SHARED_TEX_SIZE;
        // Make this the current UNPACK buffer aka PBO (Pixel Buffer Object)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, shared_buf_id);
        // Allocate data for the buffer
        glBufferData(GL_PIXEL_UNPACK_BUFFER, shared_tex_width * shared_tex_height * 4, GL_DYNAMIC_COPY);

        // Create a GL Texture
        shared_tex_id = glGenTextures();
        glBindTexture(GL_TEXTURE_2D, shared_tex_id);
        // Allocate the texture memory.
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, shared_tex_width, shared_tex_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);
        // Set filter mode
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	}

	private static int createShader(String resource, int type) throws IOException {
		int shader = glCreateShader(type);
		ByteBuffer source = resourceToByteBuffer(resource);
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
		int vshader = createShader("basic_vert.glsl", GL_VERTEX_SHADER);
		int fshader = createShader("basic_frag.glsl", GL_FRAGMENT_SHADER);
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
		basic_prog_a_position = glGetAttribLocation(program, "position");
		basic_prog_a_texCoords = glGetAttribLocation(program, "texCoords");
		basic_prog_u_cameraToView = glGetUniformLocation(program, "cameraToView");
		glUseProgram(0);
		basic_prog = program;
	}

	public static void handleResize() {
		glViewport(0, 0, window_width, window_height);
		if (window_resized) {
			window_resized = false;
			float aspect = (float) window_width / (float) window_height;
			cameraToView.identity();
			if (window_width >= window_height) {
				// aspect >= 1, set the width to -1 to 1, with larger height
				cameraToView.ortho(-1.0f, 1.0f, -1.0f / aspect, 1.0f / aspect, -1, 1);
			} else {
				// aspect < 1, set the height from -1 to 1, with larger width
				cameraToView.ortho(-1.0f * aspect, 1.0f * aspect, -1.0f, 1.0f, -1, 1);
			}
		}
	}

	public static void render() {
		glClear(GL_COLOR_BUFFER_BIT);
		glUseProgram(basic_prog);
		FloatBuffer fb = BufferUtils.createFloatBuffer(16);
		cameraToView.get(fb);
		glUniformMatrix4fv(basic_prog_u_cameraToView, false, fb);
		glBindVertexArray(verts);
		
        // connect the pbo to the texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, shared_buf_id);
        glBindTexture(GL_TEXTURE_2D, shared_tex_id);
        // Since source parameter is NULL, Data is coming from a PBO, not host memory
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, shared_tex_width, shared_tex_height, GL_BGRA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		glDrawArrays(GL_TRIANGLES, 0, 3);
		glBindVertexArray(0);
		glUseProgram(0);
	}

	public static void destroy() {
		debugProc.free();
	}
}
