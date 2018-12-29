# jmandelbrotr

A [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set) pan/zoom application powered via [Java](https://www.java.com/), [CUDA](https://developer.nvidia.com/cuda-zone) (via [JCUDA](http://www.jcuda.org/)) and [OpenGL](https://www.opengl.org/) (via [LWJGL](https://www.lwjgl.org/) and [JOML](https://github.com/JOML-CI/JOML)).  

![Screenshot](screenshot1.jpg)


## Usage

I built this using [Eclipse](https://www.eclipse.org/).  I don't really know how to properly package Java apps yet.  Hopefully, the pom.xml is enough for other Java IDEs.  Suggestions & pull requests would be considered.

An [NVIDIA](https://www.nvidia.com/) graphics card is required.  If you have a laptop with hybrid graphics, use the NVIDIA Control Panel's Manage 3D Settings panel and Program Settings tab to make Java choose the "High-performance NVIDIA processor".

Making the window smaller will speed up the calculation.

### Mouse 

Left click to pan.
Middle scroll wheel to zoom.

### Key Bindings

* __ESC__: quit
* __d/s__: (d)ouble or (s)ingle floating point precision math.  Single is default.  It is faster & less precise.
* __f__: switch to/from (f)ullscreen.
* __enter__: begin zoom out mode.  animates un-zooming.
* __1/2/3/4__: allow for N times 256 levels for determining whether in/out of mandelbrot set.
* __p__: (p)rint out some state.
* __w__: (w)rite out current screen to "save.png" -- careful, it will overwrite existing files.

LOL, I thought that glfw would eventually allow me to make a pop-up menu to control these things...but no, that will not be possible.  Enjoy the old-school keyboard shortcuts, they build character.

## License

Since I cobbled this code together basically by stringing existing example code together, I'll be giving this back to the community.

<p xmlns:dct="http://purl.org/dc/terms/" xmlns:vcard="http://www.w3.org/2001/vcard-rdf/3.0#">
  <a rel="license"
     href="http://creativecommons.org/publicdomain/zero/1.0/">
    <img src="http://i.creativecommons.org/p/zero/1.0/88x31.png" style="border-style: none;" alt="CC0" />
  </a>
  <br />
  To the extent possible under law,
  <a rel="dct:publisher"
     href="https://github.com/rogerallen/jmandelbrotr">
    <span property="dct:title">Roger Allen</span></a>
  has waived all copyright and related or neighboring rights to
  <span property="dct:title">JMandelbrotr</span>.
This work is published from:
<span property="vcard:Country" datatype="dct:ISO3166"
      content="US" about="https://github.com/rogerallen/jmandelbrotr">
  United States</span>.
</p>