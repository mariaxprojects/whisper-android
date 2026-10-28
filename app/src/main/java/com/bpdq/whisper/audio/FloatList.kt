package com.bpdq.whisper.audio

/** Growable float buffer without boxing. */
class FloatList(initialCapacity: Int = 16_000 * 30) {
    private var data = FloatArray(initialCapacity)
    var size = 0
        private set

    fun add(v: Float) {
        if (size == data.size) grow(size + 1)
        data[size++] = v
    }

    fun addAll(src: FloatArray, offset: Int = 0, length: Int = src.size) {
        if (size + length > data.size) grow(size + length)
        System.arraycopy(src, offset, data, size, length)
        size += length
    }

    private fun grow(min: Int) {
        var cap = data.size.coerceAtLeast(1024)
        while (cap < min) cap += cap / 2
        data = data.copyOf(cap)
    }

    fun toArray(): FloatArray = data.copyOf(size)
}
