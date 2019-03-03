package rogerallen.jmandelbrotr;

// started with https://www.lwjgl.org/guide
// lwjgl maven deps are at https://www.lwjgl.org/customize
// jcuda maven deps are at http://www.jcuda.org/downloads/downloads.html
// used this for inspiration https://github.com/LWJGL/lwjgl3-demos/blob/master/src/org/lwjgl/demo/opengl/textures/SimpleTexturedQuad.java

import static org.lwjgl.glfw.Callbacks.glfwFreeCallbacks;
import static org.lwjgl.glfw.GLFW.GLFW_CONTEXT_VERSION_MAJOR;
import static org.lwjgl.glfw.GLFW.GLFW_CONTEXT_VERSION_MINOR;
import static org.lwjgl.glfw.GLFW.GLFW_FALSE;
import static org.lwjgl.glfw.GLFW.GLFW_KEY_1;
import static org.lwjgl.glfw.GLFW.GLFW_KEY_2;
import static org.lwjgl.glfw.GLFW.GLFW_KEY_3;
import static org.lwjgl.glfw.GLFW.GLFW_KEY_4;
import static org.lwjgl.glfw.GLFW.GLFW_KEY_D;
import static org.lwjgl.glfw.GLFW.GLFW_KEY_ENTER;
import static org.lwjgl.glfw.GLFW.GLFW_KEY_ESCAPE;
import static org.lwjgl.glfw.GLFW.GLFW_KEY_F;
import static org.lwjgl.glfw.GLFW.GLFW_KEY_P;
import static org.lwjgl.glfw.GLFW.GLFW_KEY_S;
import static org.lwjgl.glfw.GLFW.GLFW_KEY_W;
import static org.lwjgl.glfw.GLFW.GLFW_MOUSE_BUTTON_1;
import static org.lwjgl.glfw.GLFW.GLFW_PRESS;
import static org.lwjgl.glfw.GLFW.GLFW_RELEASE;
import static org.lwjgl.glfw.GLFW.GLFW_RESIZABLE;
import static org.lwjgl.glfw.GLFW.GLFW_TRUE;
import static org.lwjgl.glfw.GLFW.GLFW_VERSION_MAJOR;
import static org.lwjgl.glfw.GLFW.GLFW_VERSION_MINOR;
import static org.lwjgl.glfw.GLFW.GLFW_VISIBLE;
import static org.lwjgl.glfw.GLFW.glfwCreateWindow;
import static org.lwjgl.glfw.GLFW.glfwDefaultWindowHints;
import static org.lwjgl.glfw.GLFW.glfwDestroyWindow;
import static org.lwjgl.glfw.GLFW.glfwGetCursorPos;
import static org.lwjgl.glfw.GLFW.glfwGetPrimaryMonitor;
import static org.lwjgl.glfw.GLFW.glfwGetVideoMode;
import static org.lwjgl.glfw.GLFW.glfwGetWindowSize;
import static org.lwjgl.glfw.GLFW.glfwInit;
import static org.lwjgl.glfw.GLFW.glfwMakeContextCurrent;
import static org.lwjgl.glfw.GLFW.glfwPollEvents;
import static org.lwjgl.glfw.GLFW.glfwSetErrorCallback;
import static org.lwjgl.glfw.GLFW.glfwSetFramebufferSizeCallback;
import static org.lwjgl.glfw.GLFW.glfwSetKeyCallback;
import static org.lwjgl.glfw.GLFW.glfwSetMouseButtonCallback;
import static org.lwjgl.glfw.GLFW.glfwSetScrollCallback;
import static org.lwjgl.glfw.GLFW.glfwSetWindowMonitor;
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

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;

import javax.imageio.ImageIO;

import org.lwjgl.BufferUtils;
import org.lwjgl.Version;
import org.lwjgl.glfw.GLFWErrorCallback;
import org.lwjgl.glfw.GLFWVidMode;
import org.lwjgl.system.MemoryStack;

import jcuda.runtime.JCuda;

public class App {

	private final String WINDOW_TITLE = "JMandelbrotr";
	private final int WINDOW_START_WIDTH = 800, WINDOW_START_HEIGHT = 800;
	private AppWindow window;

	private boolean switchFullscreen;
	private boolean isFullscreen;
	private int monitorWidth, monitorHeight;
	private int prevWindowWidth, prevWindowHeight;
	private boolean zoomOutMode;
	private boolean saveImage;
	private boolean mouseDown;
	private double mouseStartX, mouseStartY, centerStartX, centerStartY;
	DoubleBuffer mouseBufX, mouseBufY;

	public static void main(String[] args) {
		new App().run();
	}

	public void run() {
		System.out.println("JMandelbrotr");
		System.out.println("Running LWJGL " + Version.getVersion());
		System.out.println("Running GLFW " + GLFW_VERSION_MAJOR + "." + GLFW_VERSION_MINOR);
		System.out.println("Running JCuda " + JCuda.CUDART_VERSION);
		System.out.println("Press ESC to quit.");
		try {
			boolean error = init();
			if (!error) {
				loop();
				destroy();
			}
		} catch (IOException e) {
			System.err.println("ERROR: " + e);
		}
	}

	// return true if there is an error
	private boolean init() throws IOException {
		switchFullscreen = false;
		isFullscreen = false;
		zoomOutMode = false;
		saveImage = false;
		mouseBufX = BufferUtils.createDoubleBuffer(1);
		mouseBufY = BufferUtils.createDoubleBuffer(1);

		initGLFWWindow();
		initCallbacks();
		AppGL.init(window, monitorWidth, monitorHeight);
		boolean error = AppCUDA.init(window);
		return error;
	}

	private void initCallbacks() {
		// keys
		glfwSetKeyCallback(window.id(), (windowID, key, scancode, action, mods) -> {
			if (action == GLFW_PRESS) {
				switch (key) {
				case GLFW_KEY_ESCAPE:
					glfwSetWindowShouldClose(windowID, true);
					break;
				case GLFW_KEY_D:
					AppCUDA.doublePrecision = true;
					break;
				case GLFW_KEY_S:
					AppCUDA.doublePrecision = false;
					break;
				case GLFW_KEY_F:
					switchFullscreen = true;
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
		glfwSetFramebufferSizeCallback(window.id(), (windowID, width, height) -> {
			if (width > 0 && height > 0 && (window.width() != width || window.height() != height)) {
				window.width(width);
				window.height(height);
			}
		});

		// mouse up/down
		glfwSetMouseButtonCallback(window.id(), (windowID, button, action, mods) -> {
			if (button == GLFW_MOUSE_BUTTON_1) {
				if (action == GLFW_PRESS) {
					glfwGetCursorPos(windowID, mouseBufX, mouseBufY);
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
		glfwSetScrollCallback(window.id(), (windowID, xoffset, yoffset) -> {
			double zoomFactor = Math.abs(/* yoffset */ 1.1);
			if (yoffset > 0) {
				AppCUDA.zoom *= zoomFactor;
			} else {
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
		window = new AppWindow(WINDOW_START_WIDTH, WINDOW_START_HEIGHT, WINDOW_TITLE);
		long windowID = glfwCreateWindow(window.width(), window.height(), window.title(), NULL, NULL);
		if (windowID == NULL)
			throw new RuntimeException("Failed to create the GLFW window");
		window.id(windowID);

		// Center the window
		try (MemoryStack stack = stackPush()) {
			IntBuffer pWidth = stack.mallocInt(1); // int*
			IntBuffer pHeight = stack.mallocInt(1); // int*
			glfwGetWindowSize(window.id(), pWidth, pHeight);
			GLFWVidMode vidmode = glfwGetVideoMode(glfwGetPrimaryMonitor());
			monitorWidth = vidmode.width();
			monitorHeight = vidmode.height();
			glfwSetWindowPos(window.id(), (vidmode.width() - pWidth.get(0)) / 2, (vidmode.height() - pHeight.get(0)) / 2);
		}

		glfwMakeContextCurrent(window.id());
		glfwSwapInterval(1); // Enable v-sync
		glfwShowWindow(window.id());
	}

	private void update() {
		if (switchFullscreen) {
			switchFullscreen = false;
			if (isFullscreen) {
				isFullscreen = false;
				GLFWVidMode vidmode = glfwGetVideoMode(glfwGetPrimaryMonitor());
				glfwSetWindowMonitor(window.id(), 0, // no monitor, so go windowed.
						(vidmode.width() - prevWindowWidth) / 2, (vidmode.height() - prevWindowHeight) / 2,
						prevWindowWidth, prevWindowHeight, 0);
			} else {
				isFullscreen = true;
				prevWindowWidth = window.width();
				prevWindowHeight = window.height();
				GLFWVidMode vidmode = glfwGetVideoMode(glfwGetPrimaryMonitor());
				glfwSetWindowMonitor(window.id(), glfwGetPrimaryMonitor(), 0, 0, vidmode.width(), vidmode.height(),
						vidmode.refreshRate());
			}
		}

		if (zoomOutMode) {
			AppCUDA.zoom /= 1.1;
			if (AppCUDA.zoom < 0.5) {
				zoomOutMode = false;
			}
		}

		if (mouseDown) {
			glfwGetCursorPos(window.id(), mouseBufX, mouseBufY);
			double x = mouseBufX.get(0);
			double y = mouseBufY.get(0);
			double dx = x - mouseStartX;
			double dy = y - mouseStartY;
			double pixels_per_mspace;
			if (window.width() > window.height()) {
				pixels_per_mspace = window.width() * AppCUDA.zoom;
			} else {
				pixels_per_mspace = window.height() * AppCUDA.zoom;
			}
			double mspace_per_pixel = 2.0 / pixels_per_mspace;
			double center_delta_x = dx * mspace_per_pixel;
			double center_delta_y = dy * mspace_per_pixel;
			AppCUDA.centerX = centerStartX - center_delta_x;
			AppCUDA.centerY = centerStartY - center_delta_y;
		}

		if (saveImage) {
			System.out.println("write save.png\n");
			saveImage = false;
			ByteBuffer buffer = AppGL.getPixels();
			// TODO - could read into the buffer, then in another thread save the file to
			// avoid refresh delays.
			File file = new File("save.png");
			String format = "PNG"; // Example: "PNG" or "JPG"
			BufferedImage image = new BufferedImage(window.width(), window.height(), BufferedImage.TYPE_INT_RGB);

			for (int x = 0; x < window.width(); x++) {
				for (int y = 0; y < window.height(); y++) {
					int i = (x + (window.width() * y)) * 4;
					int r = buffer.get(i) & 0xFF;
					int g = buffer.get(i + 1) & 0xFF;
					int b = buffer.get(i + 2) & 0xFF;
					image.setRGB(x, window.height() - (y + 1), (0xFF << 24) | (r << 16) | (g << 8) | b);
				}
			}

			try {
				ImageIO.write(image, format, file);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	private void loop() {
		while (!glfwWindowShouldClose(window.id())) {
			glfwPollEvents();
			AppGL.handleResize();
			update();
			AppCUDA.render();
			AppGL.render();
			glfwSwapBuffers(window.id()); // swap the color buffers
		}
	}

	private void destroy() {
		glfwFreeCallbacks(window.id());
		glfwDestroyWindow(window.id());
		AppGL.destroy();
		glfwTerminate();
		glfwSetErrorCallback(null).free();
	}

}