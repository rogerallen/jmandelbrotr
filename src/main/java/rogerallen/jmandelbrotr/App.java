package rogerallen.jmandelbrotr;

// started with https://www.lwjgl.org/guide
// lwjgl maven deps are at https://www.lwjgl.org/customize
// jcuda maven deps are at http://www.jcuda.org/downloads/downloads.html
// now adding https://github.com/LWJGL/lwjgl3-demos/blob/master/src/org/lwjgl/demo/opengl/textures/SimpleTexturedQuad.java

import static org.lwjgl.glfw.Callbacks.glfwFreeCallbacks;
import static org.lwjgl.glfw.GLFW.GLFW_CONTEXT_VERSION_MAJOR;
import static org.lwjgl.glfw.GLFW.GLFW_CONTEXT_VERSION_MINOR;
import static org.lwjgl.glfw.GLFW.GLFW_FALSE;
import static org.lwjgl.glfw.GLFW.GLFW_KEY_ESCAPE;
import static org.lwjgl.glfw.GLFW.GLFW_RELEASE;
import static org.lwjgl.glfw.GLFW.GLFW_RESIZABLE;
import static org.lwjgl.glfw.GLFW.GLFW_TRUE;
import static org.lwjgl.glfw.GLFW.GLFW_VISIBLE;
import static org.lwjgl.glfw.GLFW.glfwCreateWindow;
import static org.lwjgl.glfw.GLFW.glfwDefaultWindowHints;
import static org.lwjgl.glfw.GLFW.glfwDestroyWindow;
import static org.lwjgl.glfw.GLFW.glfwGetPrimaryMonitor;
import static org.lwjgl.glfw.GLFW.glfwGetVideoMode;
import static org.lwjgl.glfw.GLFW.glfwGetWindowSize;
import static org.lwjgl.glfw.GLFW.glfwInit;
import static org.lwjgl.glfw.GLFW.glfwMakeContextCurrent;
import static org.lwjgl.glfw.GLFW.glfwPollEvents;
import static org.lwjgl.glfw.GLFW.glfwSetErrorCallback;
import static org.lwjgl.glfw.GLFW.glfwSetFramebufferSizeCallback;
import static org.lwjgl.glfw.GLFW.glfwSetKeyCallback;
import static org.lwjgl.glfw.GLFW.glfwSetWindowPos;
import static org.lwjgl.glfw.GLFW.glfwSetWindowShouldClose;
import static org.lwjgl.glfw.GLFW.glfwShowWindow;
import static org.lwjgl.glfw.GLFW.glfwSwapBuffers;
import static org.lwjgl.glfw.GLFW.glfwSwapInterval;
import static org.lwjgl.glfw.GLFW.glfwTerminate;
import static org.lwjgl.glfw.GLFW.glfwWindowHint;
import static org.lwjgl.glfw.GLFW.glfwWindowShouldClose;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.NULL;

import java.io.IOException;
import java.nio.IntBuffer;

import org.lwjgl.Version;
import org.lwjgl.glfw.GLFWErrorCallback;
import org.lwjgl.glfw.GLFWVidMode;
import org.lwjgl.system.MemoryStack;

import jcuda.runtime.JCuda;

public class App {

	private long window;
	private final String WINDOW_TITLE = "JMandelbrotr";

	public static void main(String[] args) {

		AppCUDA.test(); // FIXME

		new App().run();
	}

	public void run() {
		System.err.println("JMandelbrotr");
		System.err.println("Running LWJGL " + Version.getVersion());
		System.err.println("Running JCuda " + JCuda.CUDART_VERSION);
		System.err.println("Press ESC to quit.");
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
			if (width > 0 && height > 0 && (AppGL.window_width != width || AppGL.window_height != height)) {
				AppGL.window_width = width;
				AppGL.window_height = height;
				AppGL.window_resized = true;
			}
		});

		// init GL stuff
		AppGL.init();

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
		window = glfwCreateWindow(AppGL.window_width, AppGL.window_height, WINDOW_TITLE, NULL, NULL);
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

	private void loop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			AppGL.handleResize();
			AppGL.render();
			glfwSwapBuffers(window); // swap the color buffers
		}
	}

	private void destroy() {
		glfwFreeCallbacks(window);
		glfwDestroyWindow(window);
		AppGL.destroy();
		glfwTerminate();
		glfwSetErrorCallback(null).free();
	}

}