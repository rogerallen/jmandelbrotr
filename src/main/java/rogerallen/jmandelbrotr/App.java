package rogerallen.jmandelbrotr;

// started with https://www.lwjgl.org/guide
// lwjgl maven deps are at https://www.lwjgl.org/customize
// jcuda maven deps are at http://www.jcuda.org/downloads/downloads.html
// now adding https://github.com/LWJGL/lwjgl3-demos/blob/master/src/org/lwjgl/demo/opengl/textures/SimpleTexturedQuad.java

import static org.lwjgl.glfw.Callbacks.glfwFreeCallbacks;
import static org.lwjgl.glfw.GLFW.*;
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
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;

import org.lwjgl.BufferUtils;
import org.lwjgl.Version;
import org.lwjgl.glfw.GLFW;
import org.lwjgl.glfw.GLFWErrorCallback;
import org.lwjgl.glfw.GLFWVidMode;
import org.lwjgl.system.MemoryStack;

import jcuda.runtime.JCuda;

public class App {

	private long window;
	private final String WINDOW_TITLE = "JMandelbrotr";

	private boolean switchToFullscreen; // FIXME -- add feature
	private boolean zoomOutMode; // FIXME -- add feature
	private boolean saveImage; // FIXME -- add feature

	private boolean mouseDown;
	private double mouseStartX, mouseStartY, centerStartX, centerStartY;
	DoubleBuffer mouseBufX, mouseBufY;

	public static void main(String[] args) {
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
		switchToFullscreen = false;
		zoomOutMode = false;
		saveImage = false;
		mouseBufX = BufferUtils.createDoubleBuffer(1);
		mouseBufY = BufferUtils.createDoubleBuffer(1);
		
		initGLFWWindow();
		initCallbacks();
		AppGL.init();
		AppCUDA.init();
	}

	private void initCallbacks() {
		// keys
		glfwSetKeyCallback(window, (window, key, scancode, action, mods) -> {
			if (action == GLFW_PRESS) {
				switch (key) {
				case GLFW_KEY_ESCAPE:
					glfwSetWindowShouldClose(window, true);
					break;
				case GLFW_KEY_D:
					AppCUDA.doublePrecision = true;
					break;
				case GLFW_KEY_S:
					AppCUDA.doublePrecision = false;
					break;
				case GLFW_KEY_F:
					switchToFullscreen = true;
					break;
				case GLFW_KEY_ENTER:
					zoomOutMode = true;
					break;
				case GLFW_KEY_1:
					AppCUDA.iterMult = 1;
					break;
				case GLFW_KEY_2:
					AppCUDA.iterMult = 2;
					break;
				case GLFW_KEY_3:
					AppCUDA.iterMult = 3;
					break;
				case GLFW_KEY_4:
					AppCUDA.iterMult = 4;
					break;
				case GLFW_KEY_P:
					System.out.println("Center = " + AppCUDA.centerX + ", " + AppCUDA.centerY);
					System.out.println("Zoom = " + AppCUDA.zoom);
					break;
				case GLFW_KEY_W:
					saveImage = true;
					break;
				}
			}
		});

		// resize
		glfwSetFramebufferSizeCallback(window, (long window, int width, int height) -> {
			if (width > 0 && height > 0 && (AppGL.window_width != width || AppGL.window_height != height)) {
				AppGL.window_width = width;
				AppGL.window_height = height;
				AppGL.window_resized = true;
			}
		});

		// mouse up/down
		glfwSetMouseButtonCallback(window, (window, button, action, mods) -> {
			if (button == GLFW_MOUSE_BUTTON_1) {
				if (action == GLFW_PRESS) {
					glfwGetCursorPos(window, mouseBufX, mouseBufY);
					mouseStartX = mouseBufX.get(0);
					mouseStartY = mouseBufY.get(0);
					centerStartX = AppCUDA.centerX;
					centerStartY = AppCUDA.centerY; 
					mouseDown = true;
				} else if (action == GLFW_RELEASE) {
					mouseDown = false;
				}
			}
		});
		
		// mouse scroll 
		glfwSetScrollCallback(window, (window, xoffset, yoffset) -> {
			double zoomFactor = Math.abs(/*yoffset */ 1.1);
			if(yoffset > 0) {
				AppCUDA.zoom *= zoomFactor;
			}
			else {
				AppCUDA.zoom /= zoomFactor;
			}
		});
	}

	private void initGLFWWindow() {
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
	
	private void update() {
		// FIXME switch to fullscreen
		// FIXME zoom out mode
		if(mouseDown) {
			glfwGetCursorPos(window, mouseBufX, mouseBufY);
			double x = mouseBufX.get(0);
			double y = mouseBufY.get(0);
			double dx = x - mouseStartX;
			double dy = y - mouseStartY;
			double pixels_per_mspace;
			if (AppGL.window_width > AppGL.window_height) { 
				pixels_per_mspace = AppGL.window_width*AppCUDA.zoom;
			} else {
				pixels_per_mspace = AppGL.window_height*AppCUDA.zoom;
			}
	        double mspace_per_pixel = 2.0/pixels_per_mspace;
	        double center_delta_x = dx*mspace_per_pixel;
	        double center_delta_y = dy*mspace_per_pixel;
	        AppCUDA.centerX = centerStartX - center_delta_x;
	        AppCUDA.centerY = centerStartY - center_delta_y;
		}
		// FIXME -- save image here
	}

	private void loop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			AppGL.handleResize();
			update();			
			AppCUDA.render();
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