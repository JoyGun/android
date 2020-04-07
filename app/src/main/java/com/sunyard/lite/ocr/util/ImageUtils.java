package com.sunyard.lite.ocr.util;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Environment;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * @author llx
 * @date 2019-11-7
 */
public class ImageUtils {
    /**
     * 缩放图片
     * @param origin
     * @param width 缩放后的宽
     * @param height 缩放后的长
     * @param x0 缩放左上角x
     * @param y0
     * @param x1 缩放右下角x
     * @param y1
     * @return
     */
    public static Bitmap scaleBitmap(Bitmap origin, int width, int height, int x0, int y0 ,int x1, int y1){
        Matrix matrix = new Matrix();
        float scaleRatioX = width * 1.0f / (x1 -x0);
        float scaleRatioY = height * 1.0f / (y1 - y0);
        matrix.preScale(scaleRatioX,scaleRatioY);
        return Bitmap.createBitmap(origin,x0,y0,x1,y1,matrix,false);
    }

    public static String saveBitmap(Bitmap mBitmap,String fileName){
        File saveImage = new File(Environment.getExternalStorageDirectory(), fileName);
        try {
            if (saveImage.exists()) {
                saveImage.delete();
            }
            saveImage.createNewFile();
        }
        catch (IOException ex) {
            ex.printStackTrace();
        }
        try{
            FileOutputStream fos = new FileOutputStream(saveImage);
            mBitmap.compress(Bitmap.CompressFormat.JPEG,100,fos);
            fos.flush();
            fos.close();
        }catch (Exception e){
            e.printStackTrace();
        }
        String imageFileUrl = saveImage.getAbsolutePath();
        return imageFileUrl;
    }

    /**
     * Converts a immutable bitmap to a mutable bitmap. This operation doesn't allocates
     * more memory that there is already allocated.
     *
     * @param imgIn - Source image. It will be released, and should not be used more
     * @return a copy of imgIn, but muttable.
     */
    public static Bitmap convertToMutable(Bitmap imgIn) {
        try {
            //this is the file going to use temporally to save the bytes.
            // This file will not be a image, it will store the raw image data.
            File file = new File(Environment.getExternalStorageDirectory() + File.separator + "temp.tmp");

            //Open an RandomAccessFile
            //Make sure you have added uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"
            //into AndroidManifest.xml file
            RandomAccessFile randomAccessFile = new RandomAccessFile(file, "rw");

            // get the width and height of the source bitmap.
            int width = imgIn.getWidth();
            int height = imgIn.getHeight();
            Bitmap.Config type = imgIn.getConfig();

            //Copy the byte to the file
            //Assume source bitmap loaded using options.inPreferredConfig = Config.ARGB_8888;
            FileChannel channel = randomAccessFile.getChannel();
            MappedByteBuffer map = channel.map(FileChannel.MapMode.READ_WRITE, 0, imgIn.getRowBytes()*height);
            imgIn.copyPixelsToBuffer(map);
            //recycle the source bitmap, this will be no longer used.
            imgIn.recycle();
            System.gc();// try to force the bytes from the imgIn to be released

            //Create a new bitmap to load the bitmap again. Probably the memory will be available.
            imgIn = Bitmap.createBitmap(width, height, type);
            map.position(0);
            //load it back from temporary
            imgIn.copyPixelsFromBuffer(map);
            //close the temporary file and channel , then delete that also
            channel.close();
            randomAccessFile.close();

            // delete the temp file
            file.delete();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return imgIn;
    }

    /**
     * 旋转bitmap
     * @param bm 原图
     * @param orientationDegree 旋转角度
     * @return
     */
    public static Bitmap adjustPhotoRotation(Bitmap bm, final int orientationDegree)
    {

        Matrix m = new Matrix();
        m.setRotate(orientationDegree, (float) bm.getWidth() / 2, (float) bm.getHeight() / 2);

        try {

            Bitmap bm1 = Bitmap.createBitmap(bm, 0, 0, bm.getWidth(), bm.getHeight(), m, true);

            return bm1;

        } catch (OutOfMemoryError ex) {
        }

        return null;

    }
}
