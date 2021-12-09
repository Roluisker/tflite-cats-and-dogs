package com.tflite.cats.dogs

import android.annotation.SuppressLint
import android.content.res.AssetFileDescriptor
import android.content.res.AssetManager
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*

internal class Classifier(
    assetManager: AssetManager,
    modelPath: String,
    labelPath: String,
    private val INPUT_SIZE: Int
) {
    private val interpreter: Interpreter
    private val labelList: List<String>
    private val PIXEL_SIZE = 3
    private val IMAGE_MEAN = 0
    private val IMAGE_STD = 255.0f
    private val MAX_RESULTS = 3f
    private val THRESHOLD = 0.4f

    internal inner class Recognition(i: String, s: String, confidence: Float) {
        var id = ""
        var title = ""
        var confidence = 0f
        override fun toString(): String {
            return "Title = $title, Confidence = $confidence"
        }

        init {
            id = i
            title = s
            this.confidence = confidence
        }
    }

    @Throws(IOException::class)
    private fun loadModelFile(assetManager: AssetManager, MODEL_FILE: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(MODEL_FILE)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    @Throws(IOException::class)
    private fun loadLabelList(assetManager: AssetManager, labelPath: String): List<String> {
        val labelList: MutableList<String> = ArrayList()
        val reader = BufferedReader(InputStreamReader(assetManager.open(labelPath)))

        reader.forEachLine {  label ->
            labelList.add(label)
        }

        reader.close()
        return labelList
    }

    /**
     * Returns the result after running the recognition with the help of interpreter
     * on the passed bitmap
     */
    fun recognizeImage(bitmap: Bitmap?): List<Recognition> {
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap!!, INPUT_SIZE, INPUT_SIZE, false)
        val byteBuffer = convertBitmapToByteBuffer(scaledBitmap)
        val result = Array(1) {
            FloatArray(
                labelList.size
            )
        }
        interpreter.run(byteBuffer, result)
        return getSortedResultFloat(result)
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer: ByteBuffer
        byteBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until INPUT_SIZE) {
            for (j in 0 until INPUT_SIZE) {
                val `val` = intValues[pixel++]
                byteBuffer.putFloat(((`val` shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                byteBuffer.putFloat(((`val` shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                byteBuffer.putFloat(((`val` and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
            }
        }
        return byteBuffer
    }

    @SuppressLint("DefaultLocale")
    private fun getSortedResultFloat(labelProbArray: Array<FloatArray>): List<Recognition> {
        val pq = PriorityQueue(
            MAX_RESULTS.toInt(),
            object : Comparator<Recognition?> {
                override fun compare(lhs: Recognition?, rhs: Recognition?): Int {
                    return java.lang.Float.compare(rhs!!.confidence, lhs!!.confidence)
                }
            })
        for (i in labelList.indices) {
            val confidence = labelProbArray[0][i]
            if (confidence > THRESHOLD) {
                pq.add(
                    Recognition(
                        "" + i,
                        if (labelList.size > i) labelList[i] else "unknown",
                        confidence
                    )
                )
            }
        }
        val recognitions = ArrayList<Recognition>()
        val recognitionsSize = Math.min(pq.size.toFloat(), MAX_RESULTS).toInt()
        for (i in 0 until recognitionsSize) {
            pq.poll()?.let { recognitions.add(it) }
        }
        return recognitions
    }

    init {
        val options = Interpreter.Options()
        options.setNumThreads(5)
        options.setUseNNAPI(true)
        interpreter = Interpreter(loadModelFile(assetManager, modelPath), options)
        labelList = loadLabelList(assetManager, labelPath)
    }
}