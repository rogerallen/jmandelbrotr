package rogerallen.jmandelbrotr;

public class AppWindow {
	private long id;
	private int width;
	private int height;
	private String title;
	private boolean resized;

	public AppWindow(int width, int height, String title) {
		this.id = -1;
		this.width  = width;
		this.height = height;
		this.title  = title;
		this.resized = true;
	}

	public long id() {
		return id;
	}
	public void id(long x) {
		this.id = x;
	}

	public int width() {
		return width;
	}
	public void width(int x) {
		this.width = x;
		this.resized = true;
	}

	public int height() {
		return height;
	}
	public void height(int x) {
		this.height = x;
		this.resized = true;
	}

	public String title() {
		return title;
	}
	public void title(String x) {
		this.title = x;
	}
	
	public boolean resized() {
		return resized;
	}
	public void resizeHandled() {
		this.resized = false;
	}
	
	@Override
	public String toString() {
		return "AppWindow: "+title+"\n  id="+id+" w="+width+" h="+height+" rs="+resized;
	}

}
