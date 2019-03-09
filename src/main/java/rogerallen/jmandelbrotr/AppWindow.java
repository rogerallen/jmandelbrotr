package rogerallen.jmandelbrotr;

/**
 * A window utility class to hold some window-related state.
 * 
 * @author rallen
 *
 */

public class AppWindow {
    private long id;
    private int width;
    private int height;
    private String title;
    private boolean resized;

    /**
     * Constructor
     * 
     * @param width  of window
     * @param height of window
     * @param title  of window
     */
    public AppWindow(int width, int height, String title) {
        this.id = -1;
        this.width = width;
        this.height = height;
        this.title = title;
        this.resized = true;
    }

    /**
     * @return window id
     */
    public long id() {
        return id;
    }

    /**
     * set window id
     * 
     * @param x is the new window id
     */
    public void id(long x) {
        this.id = x;
    }

    /**
     * @return window width
     */
    public int width() {
        return width;
    }

    /**
     * set window width (and adjust resized state)
     * 
     * @param x is the new width
     */
    public void width(int x) {
        this.width = x;
        this.resized = true;
    }

    /**
     * @return window height
     */
    public int height() {
        return height;
    }

    /**
     * set window height
     * 
     * @param x is the new height
     */
    public void height(int x) {
        this.height = x;
        this.resized = true;
    }

    /**
     * @return window title
     */
    public String title() {
        return title;
    }

    /**
     * @return true if window has been resized & not handled
     */
    public boolean resized() {
        return resized;
    }

    /**
     * handle resize event, clear resized state.
     */
    public void resizeHandled() {
        this.resized = false;
    }

    /**
     * Debug helper
     */
    @Override
    public String toString() {
        return "AppWindow: " + title + "\n  id=" + id + " w=" + width + " h=" + height + " rs=" + resized;
    }

}
