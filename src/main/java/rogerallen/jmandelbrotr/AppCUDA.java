package rogerallen.jmandelbrotr;

import jcuda.Pointer;
import jcuda.runtime.JCuda;

public class AppCUDA {

	public static void test() {
		// Simplest, stupidest proof cuda works. FIXME / TODO
		Pointer pointer = new Pointer();
		JCuda.cudaMalloc(pointer, 4);
		System.out.println("Pointer: " + pointer);
		JCuda.cudaFree(pointer);
	}

}
