package rogerallen.jmandelbrotr;

/*
 * Some portions lifted from https://github.com/LWJGL/lwjgl3-demos/blob/master/src/org/lwjgl/demo/opengl/util/DemoUtils.java
 * and subject to this license.
 * 
 * Copyright © 2012-present Lightweight Java Game Library All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer. Redistributions in binary
 * form must reproduce the above copyright notice, this list of conditions and
 * the following disclaimer in the documentation and/or other materials provided
 * with the distribution. Neither the name Lightweight Java Game Library nor the
 * names of its contributors may be used to endorse or promote products derived
 * from this software without specific prior written permission. 
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
 * ANY EXPRESSOR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 */

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

import org.lwjgl.BufferUtils;

/**
 * Static Java Utility functions.
 * 
 * @author rallen
 *
 */
public class AppUtils {

    /**
     * resize an existing ByteBuffer & return the new buffer.
     * 
     * @param buffer      - existing buffer.
     * @param newCapacity the size of the new buffer.
     * @return new buffer
     */
    public static ByteBuffer resizeBuffer(ByteBuffer buffer, int newCapacity) {
        ByteBuffer newBuffer = BufferUtils.createByteBuffer(newCapacity);
        buffer.flip();
        newBuffer.put(buffer);
        return newBuffer;
    }

    /**
     * read a named File or Resource and return it as a ByteBuffer
     * 
     * @param resource is the name of the resource
     * @return resource read in as a ByteBuffer
     * @throws IOException
     */
    public static ByteBuffer ioResourceToByteBuffer(String resource) throws IOException {
        int bufferSize = 8192;
        ByteBuffer buffer;
        URL url = Thread.currentThread().getContextClassLoader().getResource(resource);
        File file = new File(url.getFile());
        if (file.isFile()) {
            FileInputStream fis = new FileInputStream(file);
            FileChannel fc = fis.getChannel();
            buffer = fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size());
            fc.close();
            fis.close();
        } else {
            buffer = BufferUtils.createByteBuffer(bufferSize);
            InputStream source = url.openStream();
            if (source == null)
                throw new FileNotFoundException(resource);
            try {
                byte[] buf = new byte[8192];
                while (true) {
                    int bytes = source.read(buf, 0, buf.length);
                    if (bytes == -1)
                        break;
                    if (buffer.remaining() < bytes)
                        buffer = AppUtils.resizeBuffer(buffer, buffer.capacity() * 2);
                    buffer.put(buf, 0, bytes);
                }
                buffer.flip();
            } finally {
                source.close();
            }
        }
        return buffer;
    }

}
