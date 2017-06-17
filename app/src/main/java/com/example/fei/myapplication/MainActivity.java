package com.example.fei.myapplication;

import android.graphics.PointF;
import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.TextView;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

import java.io.File;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity implements View.OnTouchListener {
    private static final String TAG = MainActivity.class.getSimpleName();
    private static final int PIXEL_WIDTH = 28;
    private TextView mPredictDigit;
    private TextView mRawResult;

    private float mLastX;
    private float mLastY;
    private DrawModel mModel;
    private DrawView mDrawView;
    private PointF mTmpPiont = new PointF();

    @SuppressWarnings("SuspiciousNameCombination")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mModel = new DrawModel(PIXEL_WIDTH, PIXEL_WIDTH);
        mDrawView = (DrawView) findViewById(R.id.view_draw);
        mRawResult = (TextView) findViewById(R.id.raw_result);
        mPredictDigit = (TextView)findViewById(R.id.predict_digit);

        mDrawView.setModel(mModel);
        mDrawView.setOnTouchListener(this);

        View detectButton = findViewById(R.id.button_detect);
        detectButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onDetectClicked();
            }
        });

        View clearButton = findViewById(R.id.button_clear);
        clearButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onClearClicked();
            }
        });
    }

    @Override
    protected void onResume() {
        mDrawView.onResume();
        super.onResume();
    }

    @Override
    protected void onPause() {
        mDrawView.onPause();
        super.onPause();
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        int action = event.getAction() & MotionEvent.ACTION_MASK;
        if (action == MotionEvent.ACTION_DOWN) {
            processTouchDown(event);
            return true;

        } else if (action == MotionEvent.ACTION_MOVE) {
            processTouchMove(event);
            return true;

        } else if (action == MotionEvent.ACTION_UP) {
            processTouchUp();
            return true;
        }
        return false;
    }

    private void processTouchDown(MotionEvent event) {
        mLastX = event.getX();
        mLastY = event.getY();
        mDrawView.calcPos(mLastX, mLastY, mTmpPiont);
        float lastConvX = mTmpPiont.x;
        float lastConvY = mTmpPiont.y;
        mModel.startLine(lastConvX, lastConvY);
    }

    private void processTouchMove(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        mDrawView.calcPos(x, y, mTmpPiont);
        float newConvX = mTmpPiont.x;
        float newConvY = mTmpPiont.y;
        mModel.addLineElem(newConvX, newConvY);

        mLastX = x;
        mLastY = y;
        mDrawView.invalidate();
    }

    private void processTouchUp() {
        mModel.endLine();
    }

    private void onDetectClicked() {
        int pixels[] = mDrawView.getPixelData();
        int n = pixels.length;

        new DigitDetector().execute(pixels);
    }

    private void onClearClicked() {
        mModel.clear();
        mDrawView.reset();
        mDrawView.invalidate();

        mPredictDigit.setText("");
        mRawResult.setText("");
    }

    private class DigitDetector extends AsyncTask<int[], Integer, DataBuffer> {
        protected DataBuffer doInBackground(int[]... pixels) {
            File root = android.os.Environment.getExternalStorageDirectory();

            //the model has to be pretrained in local machine and manually copied to Android.
            //TODO load the model when starting app so do not have repeating loading it.
            File locationToSave = new File(root.getAbsolutePath() + "/storage/emulated/0/" + "my_multi_layer_network.zip");

            int[] passed = pixels[0];
            MultiLayerNetwork restored;
            try {
                Log.d(TAG, "start serialize");
                restored = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
            }
            catch (Exception e) {
                e.printStackTrace();
                return null;
            }

            float[] data = new float[passed.length];
            for (int i = 0; i < data.length; i++) {
                data[i] = (float) passed[i];
            }

            Log.d(TAG, Arrays.toString(data));

            INDArray input = new NDArray(data);
            INDArray output = restored.output(input); //get the networks prediction

            return output.data();
        }

        protected void onProgressUpdate(Integer... progress) {

        }

        protected void onPostExecute(DataBuffer result) {
            if (result != null) {
                Log.d(TAG, result.toString());
            }
            else {
                Log.d(TAG, "result is null");
                return;
            }

            mRawResult.setText(result.toString());

            float[] probs = result.getFloatsAt(0L, (int) result.length());
            int maxIndex = -1;
            float maxProb = -1;
            for (int i = 0; i < probs.length; i++) {
                if (probs[i] > maxProb) {
                    maxProb = probs[i];
                    maxIndex = i;
                }
            }

            mPredictDigit.setText("Predicted: " + maxIndex + ", Prob: " + (maxProb * 100) + "%");
        }
    }
}
