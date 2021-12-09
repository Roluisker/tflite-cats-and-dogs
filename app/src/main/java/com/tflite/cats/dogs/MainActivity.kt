package com.tflite.cats.dogs

import android.graphics.drawable.BitmapDrawable
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.ImageView
import android.widget.Toast
import kotlinx.android.synthetic.main.activity_main.*
import java.io.IOException

class MainActivity : AppCompatActivity(), View.OnClickListener {

    private val mInputSize = 224
    private val mModelPath = "converted_model.tflite"
    private val mLabelPath = "label.txt"
    private var classifier: Classifier? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        try {
            initClassifier()
            iv_1.setOnClickListener(this)
            iv_2.setOnClickListener(this)
            iv_3.setOnClickListener(this)
            iv_4.setOnClickListener(this)
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    @Throws(IOException::class)
    private fun initClassifier() {
        classifier = Classifier(assets, mModelPath, mLabelPath, mInputSize)
    }

    override fun onClick(view: View) {
        val bitmap = ((view as ImageView).drawable as BitmapDrawable).bitmap

        val result: List<Classifier.Recognition> = classifier!!.recognizeImage(bitmap)

        Toast.makeText(this, result[0].toString(), Toast.LENGTH_SHORT).show()
    }
}