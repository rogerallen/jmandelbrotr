package rogerallen.jmandelbrotr;

import java.io.File;
import java.io.FileInputStream;
//import java.io.FileNotFoundException;

// started with https://www.lwjgl.org/guide
// lwjgl maven deps are at https://www.lwjgl.org/customize
// jcuda maven deps are at http://www.jcuda.org/downloads/downloads.html
// now adding https://github.com/LWJGL/lwjgl3-demos/blob/master/src/org/lwjgl/demo/opengl/textures/SimpleTexturedQuad.java

import java.io.IOException;
//import java.io.InputStream;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.channels.FileChannel;

import org.lwjgl.*;
import org.lwjgl.glfw.*;
import org.lwjgl.opengl.*;
import org.lwjgl.system.*;

import org.joml.Matrix4f;

import static org.lwjgl.glfw.Callbacks.*;
import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.opengl.GL11.*;
import static org.lwjgl.opengl.GL15.*;
import static org.lwjgl.opengl.GL20.*;
import static org.lwjgl.opengl.GL30.*;
import static org.lwjgl.stb.STBImage.*;
import static org.lwjgl.system.MemoryStack.*;
import static org.lwjgl.system.MemoryUtil.*;

import jcuda.Pointer;
import jcuda.runtime.*;

public class App {

	private long window;
	private final String WINDOW_TITLE = "JMandelbrotr";
	private int window_width = 800;
	private int window_height = 800;
	private boolean window_resized = false;

	private int verts;
	private int basic_prog;
	private int basic_prog_a_position;
	private int basic_prog_a_texCoords;
	private int basic_prog_u_cameraToView;

	private Matrix4f cameraToView = new Matrix4f();

	private GLCapabilities caps;
	private Callback debugProc;

	public static void main(String[] args) {

		// Simplest, stupidest proof cuda works. FIXME / TODO
		Pointer pointer = new Pointer();
		JCuda.cudaMalloc(pointer, 4);
		System.out.println("Pointer: " + pointer);
		JCuda.cudaFree(pointer);

		new App().run();
	}

	public void run() {
		System.out.println("JMandelbrotr");
		System.out.println("Running LWJGL " + Version.getVersion());
		System.out.println("Running JCuda " + JCuda.CUDART_VERSION);
		System.out.println("Press ESC to quit.");
		try {
			init();
			loop();
			destroy();
		} catch (IOException e) {
			System.err.println("ERROR: " + e);
		}
	}

	private void init() throws IOException {
		// init GLFW stuff
		initWindow();

		// callbacks
		glfwSetKeyCallback(window, (window, key, scancode, action, mods) -> {
			if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE) {
				glfwSetWindowShouldClose(window, true);
			}
		});
		glfwSetFramebufferSizeCallback(window, (long window, int width, int height) -> {
			if (width > 0 && height > 0 && (App.this.window_width != width || App.this.window_height != height)) {
				App.this.window_width = width;
				App.this.window_height = height;
				App.this.window_resized = true;
			}
		});

		// init GL stuff
		caps = GL.createCapabilities();
		debugProc = GLUtil.setupDebugMessageCallback();
		glClearColor(1.0f, 1.0f, 0.5f, 0.0f);
		initTexture();
		initProgram();
		initVerts();

	}

	private void initWindow() {
		GLFWErrorCallback.createPrint(System.err).set();

		if (!glfwInit())
			throw new IllegalStateException("Unable to initialize GLFW");

		glfwDefaultWindowHints();
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		// Create the window
		window = glfwCreateWindow(window_width, window_height, WINDOW_TITLE, NULL, NULL);
		if (window == NULL)
			throw new RuntimeException("Failed to create the GLFW window");

		// Center the window
		try (MemoryStack stack = stackPush()) {
			IntBuffer pWidth = stack.mallocInt(1); // int*
			IntBuffer pHeight = stack.mallocInt(1); // int*
			glfwGetWindowSize(window, pWidth, pHeight);
			GLFWVidMode vidmode = glfwGetVideoMode(glfwGetPrimaryMonitor());
			glfwSetWindowPos(window, (vidmode.width() - pWidth.get(0)) / 2, (vidmode.height() - pHeight.get(0)) / 2);
		}

		glfwMakeContextCurrent(window);
		glfwSwapInterval(1); // Enable v-sync
		glfwShowWindow(window);
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
	private void initVerts() {
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

	private void initTexture() throws IOException {
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
	}

	private int createShader(String resource, int type) throws IOException {
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

	private void initProgram() throws IOException {
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

	private void handleResize() {
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

	private void render() {
		glClear(GL_COLOR_BUFFER_BIT);
		glUseProgram(basic_prog);
		FloatBuffer fb = BufferUtils.createFloatBuffer(16);
		cameraToView.get(fb);
		glUniformMatrix4fv(basic_prog_u_cameraToView, false, fb);
		glBindVertexArray(verts);
		glDrawArrays(GL_TRIANGLES, 0, 3);
		glBindVertexArray(0);
		glUseProgram(0);
	}

	private void loop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			handleResize();
			render();
			glfwSwapBuffers(window); // swap the color buffers
		}
	}

	private void destroy() {
		glfwFreeCallbacks(window);
		glfwDestroyWindow(window);
		debugProc.free();
		glfwTerminate();
		glfwSetErrorCallback(null).free();
	}

}