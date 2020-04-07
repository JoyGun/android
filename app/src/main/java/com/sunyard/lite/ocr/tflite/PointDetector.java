package com.sunyard.lite.ocr.tflite;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Environment;

import com.sunyard.lite.ocr.MainActivity;
import com.sunyard.lite.ocr.constant.IdCardRegConstant;
import com.sunyard.lite.ocr.util.ImageUtils;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.engine.OpenCVEngineInterface;
import org.opencv.imgproc.Imgproc;
import org.opencv.osgi.OpenCVInterface;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.classification.tflite.Classifier;
import org.tensorflow.lite.examples.classification.tflite.ClassifierFloatMobileNet;
import org.tensorflow.lite.examples.classification.tflite.ClassifierQuantizedMobileNet;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileLock;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.xml.transform.Source;

import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE;
import static org.opencv.imgproc.Imgproc.INTER_LINEAR;
import static org.opencv.imgproc.Imgproc.MORPH_CLOSE;
import static org.opencv.imgproc.Imgproc.MORPH_OPEN;
import static org.opencv.imgproc.Imgproc.MORPH_RECT;
import static org.opencv.imgproc.Imgproc.RETR_TREE;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY;
import static org.opencv.imgproc.Imgproc.THRESH_OTSU;
import static org.opencv.imgproc.Imgproc.WARP_INVERSE_MAP;
import static org.opencv.imgproc.Imgproc.getStructuringElement;
import static org.opencv.imgproc.Imgproc.morphologyEx;

/**
 * 身份证四个角检测器
 */

public class PointDetector {
    private MainActivity mainActivity = null;

    private MappedByteBuffer tfliteModel;

    protected Interpreter tflite;

    private TensorImage inputImageBuffer;

    /** Output probability TensorBuffer. */
    private final TensorBuffer outputProbabilityBuffer;
    /** Processer to apply post processing of the output probability. */
    private final TensorProcessor probabilityProcessor;

    private final int imageSizeX = 800;
    private final int imageSizeY = 600;

    public static PointDetector create(Activity activity, Classifier.Model model, Classifier.Device device, int numThreads,MainActivity mainActivity)
            throws IOException {
            return new PointDetector(activity, device, numThreads,mainActivity);
    }

    /**
     * 创建模型的对象
     * @param activity
     * @param device
     * @param numThreads
     */
    protected PointDetector(Activity activity, Classifier.Device device,int numThreads,MainActivity mainActivity) throws IOException {
        this.mainActivity = mainActivity;
        tfliteModel = FileUtil.loadMappedFile(activity, getModelPath());
        Interpreter.Options tfOptions = new Interpreter.Options();
        tfOptions.setNumThreads(4);
        //tfOptions.addDelegate(new GpuDelegate());
        tflite = new Interpreter(tfliteModel,tfOptions);
        int imageTensorIndex = 0;
        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();
        int probabilityTensorIndex = 0;
        int[] probabilityShape =
                tflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
        DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();
        // Creates the output tensor and its processor.
        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);
        // Creates the post processor for the output probability.
        probabilityProcessor = new TensorProcessor.Builder().build();
        // Creates the input tensor.
        inputImageBuffer = new TensorImage(imageDataType);
    }

    private String getModelPath() {
        return "tf_detect_model_v12_0921.tflite";
    }

    public Map detectImage(final Bitmap bitmap,int regSide) {
        Bitmap bitmapArea = null;
        List<Bitmap> bitmapList = new ArrayList<>();
        int resizeHeight = 315;
        int resizeWidth = 500;

//        int resizeHeight = bitmap.getHeight();
//        int resizeWidth = bitmap.getWidth();

        Map resultMap = new HashMap();

        inputImageBuffer = loadImage(bitmap,0);
        tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
        //return tensor 25*19*4 means 4 points
        //convert tensor point to 800*600 point
        //透视变化转换原图
//        perspectiveChange(0,0,null);
        List<Float> floatList = new ArrayList<>();
        for(int i = 0;i<outputProbabilityBuffer.getFloatArray().length;i++){
            floatList.add(outputProbabilityBuffer.getFloatArray()[i]);
        }
        //获取800*600下的四个坐标点
        //Point[] points = convertTensorPoint2OriginImage(floatList);
        Point[] points = dsnt(floatList);
        if(isRect(points)){
            /**
             * 找到四个顶点后，用王毅的逻辑，透视->canny->水平投影->找文字区域
             */
            //用透视变化矫正
            Mat mat = new Mat();
//            //使用opencv得方法resize
//            Utils.bitmapToMat(bitmap, mat);
//            Imgproc.resize(mat,mat,new Size(800,600));
            Utils.bitmapToMat(ImageUtils.scaleBitmap(bitmap,800,600,0,0,bitmap.getWidth(),bitmap.getHeight()),mat);
            mat = getWarPerspective(mat, points);
            //resize mat to 500*315
            Bitmap resizeBitmap = Bitmap.createBitmap(mat.width(),mat.height(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mat,resizeBitmap);
            //获取文本区域
            Bitmap bitmap1 = ImageUtils.scaleBitmap(resizeBitmap,resizeWidth,resizeHeight,0,0,resizeBitmap.getWidth(),resizeBitmap.getHeight());
            //创建一张图用于绘制文本范围
            Bitmap bitmapDraw = ImageUtils.scaleBitmap(resizeBitmap,resizeWidth,resizeHeight,0,0,resizeBitmap.getWidth(),resizeBitmap.getHeight());
            if(regSide == IdCardRegConstant.FRONT_SIDE){
                List<PointDetector.RangesArray> rangesArrays = extractTextArea2(bitmap1,0,0,(int)(0.4*bitmap1.getWidth()),bitmap1.getHeight());
                //画出文本区域
                drawTextRect(bitmapDraw,new ArrayList<>());
                //判定为号码区域的高度阈值  0.82-0.9 因为没有用区域限定所以拉低号码区域的下限，确保不会有干扰
                int numberThreshold = (int)(0.82*resizeHeight);
                //如果出现少数民族文字干扰，文本区域大小会变大
                int minorityThreshold = 40;
                //姓名切片中“姓名”二字干扰太大，所以这里切除，认为最后一个切片为姓名切片
                int nowAreaIndex = 1;
                for(RangesArray rangesArray:rangesArrays){
                    if(nowAreaIndex == rangesArrays.size()){
                        bitmapArea = Bitmap.createBitmap(bitmap1,(int)(0.16*resizeWidth),rangesArray.start,(int)((0.67-0.16)*resizeWidth),rangesArray.end - rangesArray.start);
                        bitmapList.add(bitmapArea);
                    }else{
                        if(rangesArray.end<numberThreshold){
                            bitmapArea = Bitmap.createBitmap(bitmap1,(int)(0.*resizeWidth),rangesArray.start,(int)((0.67-0.)*resizeWidth),rangesArray.end - rangesArray.start);
                            bitmapList.add(bitmapArea);
                            //ImageUtils.saveBitmap(bitmapArea,Math.random()+".jpg");
                        }else{
                            //稍微大一点,有些把前面的字加入会影响结果
                            if((rangesArray.end - rangesArray.start) > minorityThreshold){
                                rangesArray.start += 10;
                                //rangesArray.end += 5;
                            }
                            bitmapArea = Bitmap.createBitmap(bitmap1,(int)(0.3*resizeWidth),rangesArray.start ,(int)((1 - 0.3)*resizeWidth),resizeHeight - rangesArray.end >= 0?(rangesArray.end - rangesArray.start):(resizeHeight - rangesArray.start));
                            bitmapList.add(bitmapArea);
                        }
                        nowAreaIndex++;
                    }
                }
                resultMap.put("textArea",bitmapList);
                resultMap.put("isRect",true);
                //resultMap = posRequire(resultMap);
            }else if(regSide == IdCardRegConstant.BACK_SIDE){
                //背面切片
                List<PointDetector.RangesArray> rangesArrays = extractTextArea2(bitmap1,(int)(0.16*bitmap1.getWidth()),(int)(0.5*bitmap1.getHeight()),(int)(0.84*bitmap1.getWidth()),(int)(0.5*bitmap1.getHeight()));
                //画出文本区域
                drawTextRect(bitmap1,new ArrayList<>());
                for(RangesArray rangesArray:rangesArrays){
                    bitmapArea = Bitmap.createBitmap(bitmap1,(int)(0.16*resizeWidth),rangesArray.start,(int)((1-0.16)*resizeWidth),rangesArray.end - rangesArray.start);
                    bitmapList.add(bitmapArea);
                }
                resultMap.put("textArea",bitmapList);
                resultMap.put("isRect",true);
            }
        }else{
            //用透视变化矫正
            Mat mat = new Mat();
            Utils.bitmapToMat(ImageUtils.scaleBitmap(bitmap,800,600,0,0,bitmap.getWidth(),bitmap.getHeight()),mat);
            mat = getWarPerspective(mat, points);
            //resize mat to 500*315
            Bitmap resizeBitmap = Bitmap.createBitmap(mat.width(),mat.height(), Bitmap.Config.ARGB_8888);
            //形态学提取文本区域
            Mat dilate = extractTextArea(mat);
            List<RotatedRect> regions = findTextRegion(dilate);
            //在mat上画出来看一下
            Mat matArea = null;
            //Bitmap bitmapArea = null;
            List<Bitmap> textAreas = new ArrayList<>();
            for(RotatedRect rotatedRect:regions){
                Point[] p = new Point[4];
                rotatedRect.points(p);
//            for(int j=0;j<=3;j++){
//                Imgproc.line(mat,p[j],p[(j+1)%4],new  Scalar(0,255,0),1);
//            }
                Point[] newPs = new Point[4];
                if(rotatedRect.angle<-45){
                    //寻找外接矩阵的时候会有可能会倒转，也不知道为什么
                    newPs[0] = p[2];
                    newPs[1] = p[3];
                    newPs[2] = p[1];
                    newPs[3] = p[0];
                }else{
                    newPs[0] = p[1];
                    newPs[1] = p[2];
                    newPs[2] = p[0];
                    newPs[3] = p[3];
                }
                newPs = expandTextArea(newPs,1f,resizeBitmap.getHeight());
                //System.out.println(p[0].x+","+p[0].y+":"+p[1].x+","+p[1].y+":"+p[2].x+","+p[2].y+":"+p[3].x+","+p[3].y+":");
                matArea = getWarPerspective(mat,newPs);
                bitmapArea = Bitmap.createBitmap(matArea.width(),matArea.height(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(matArea,bitmapArea);
                textAreas.add(bitmapArea);
                //ImageUtils.saveBitmap(bitmapArea,Math.random()+".jpg");
            }
            //saveMat(mat,"mat.jpg");
            //返回透视变化的文本区域
            resultMap.put("textArea",textAreas);
            resultMap.put("isRect",false);
        }
        return resultMap;
    }

    /**
     * pos上只需要姓名和号码字段，故选取前后两个字段返回
     * @param resultMap 原始的切图
     * @return pos需要的字段的可能切图
     */
    private Map posRequire(Map resultMap) {
        List postBitmapList = new ArrayList();
        ArrayList bitmapList = (ArrayList) resultMap.get("textArea");
        if(bitmapList.size()>4){
            postBitmapList = new ArrayList(Arrays.asList(
                    bitmapList.get(0),
                    //bitmapList.get(1),
                    //bitmapList.get(bitmapList.size() - 2),
                    bitmapList.get(bitmapList.size() - 1)
            ));
            resultMap.put("textArea",postBitmapList);
        }
        return resultMap;
    }

    /**
     * 画出边框
     * @param bitmap1 resize后的图片
     * @param rangesArrays 文本区域的高度
     */
    private void drawTextRect(Bitmap bitmap1, List<RangesArray> rangesArrays) {
        Mat mat = new Mat();
        Utils.bitmapToMat(bitmap1,mat);
        for(RangesArray rangesArray:rangesArrays){
            Point p1,p2,p3,p4;
            p1 = new Point(0,rangesArray.start);
            p2 = new Point(bitmap1.getWidth(),rangesArray.start);
            p3 = new Point(0,rangesArray.end);
            p4 = new Point(bitmap1.getWidth(),rangesArray.end);
            Imgproc.line(mat,p1,p2,new  Scalar(0,255,0),1);
            Imgproc.line(mat,p2,p4,new  Scalar(0,255,0),1);
            Imgproc.line(mat,p3,p4,new  Scalar(0,255,0),1);
            Imgproc.line(mat,p3,p1,new  Scalar(0,255,0),1);
        }
        Utils.matToBitmap(mat,bitmap1);
        mainActivity.canny.setImageBitmap(bitmap1);
    }

    /**
     * 判断四个点是否能组成大体的矩形
     * @param points 四点顺序左上、右上、左下、右下
     * @return
     */
    public Boolean isRect(Point[] points){
        //todo
        return true;
    }

    /**
     *
     * @param bitmap 原始图片
     * @param cannyXCoordinate canny处理部分的x坐标
     * @param cannyYCoordinate canny处理部分的y坐标
     * @param cannyXWidth canny处理部分宽度
     * @param cannyYHeight canny处理部分长度
     * @return
     */
    private List<PointDetector.RangesArray> extractTextArea2(Bitmap bitmap,int cannyXCoordinate, int cannyYCoordinate,int cannyXWidth,int cannyYHeight) {
        int minW = 10;
        int maxW = 30;
        int h = bitmap.getHeight();
        int w = bitmap.getWidth();
        List<Integer> whiteCntList = new ArrayList();
        List<PointDetector.RangesArray> rangesArrayList = new ArrayList<>();
        bitmap = Bitmap.createBitmap(bitmap,cannyXCoordinate,cannyYCoordinate,cannyXWidth,cannyYHeight,new Matrix(),false);
        Mat resizeMat = new Mat();
        Utils.bitmapToMat(bitmap,resizeMat);
        //saveMat(resizeMat,"resizeMat.jpg");
        //转成灰度图
        Mat gray = new Mat();
        Imgproc.cvtColor(resizeMat,gray,Imgproc.COLOR_RGB2GRAY);
        //saveMat(gray,"gray.jpg");
        //canny算子，求x方向的梯度
        Mat canny = new Mat();
        Imgproc.Canny(gray,canny,60,180);

//        Bitmap bitmapCanny = Bitmap.createBitmap(canny.width(),canny.height(), Bitmap.Config.ARGB_8888);
//        Utils.matToBitmap(canny,bitmapCanny);
//        mainActivity.blur.setImageBitmap(bitmapCanny);
        //saveMat(canny,"canny.jpg");
        //膨胀
        Mat elementDilate = Imgproc.getStructuringElement(MORPH_RECT,new Size(15,3));  //24,4
        Mat dilate1 = new Mat();
        Imgproc.dilate(canny,dilate1,elementDilate,new Point(-1,-1),1);
        Utils.matToBitmap(dilate1,bitmap);
        //saveMat(dilate1,"dilate1.jpg");
        //水平投影
        for(int i=0;i<cannyYHeight;i++){
            int count = 0;
            for(int j = 0;j<cannyXWidth;j++){
                int value = bitmap.getPixel(j,i);
                if(value == -1){
                    count++;
                }
            }
            if(count>=15){
                whiteCntList.add(i+cannyYCoordinate);
            }
        }
        //求出区间，各个文字切片的范围
        int start = 0;
        int end = 0;
        int peakRange = 1;
        if(whiteCntList.size()>2){
            RangesArray rangesArray = null;
            for(int i = 0;i<whiteCntList.size()-1;i++){
                //当相邻的两个索引相差10个以上的时候说明是另一个切片的开始
                if(whiteCntList.get(i+1) - whiteCntList.get(i)>1){
                    end = i;
                    peakRange = 0;
                    //过滤宽度过小的
                    int rangeWidth = whiteCntList.get(end) - whiteCntList.get(start);
                    if(rangeWidth>minW&&rangeWidth<maxW){
                        rangesArrayList.add(new RangesArray(whiteCntList.get(start),whiteCntList.get(end)));
                    }
                    start = i+1;
                }else{
                    peakRange++;
                    if(peakRange>maxW){
                        rangesArrayList.add(new RangesArray(whiteCntList.get(start),whiteCntList.get(i)));
                        start = i+1;
                        peakRange = 0;
                    }
                }
                if(i == whiteCntList.size()-2){
                    end = whiteCntList.size()-1;
                    int rangeWidth = whiteCntList.get(end) - whiteCntList.get(start);
                    if(rangeWidth>minW&&rangeWidth<maxW){
                        rangesArrayList.add(new RangesArray(whiteCntList.get(start),whiteCntList.get(end)));
                    }
                }
            }
        }
        //上下扩大五个像素
        rangesArrayList = expandTextRange(5,h,rangesArrayList);
        return rangesArrayList;
    }

    /**
     *
     * @param i 扩大的像素值
     * @param h 原图高
     * @param rangesArrayList 文本范围
     * @return
     */
    private List<RangesArray> expandTextRange(int i, int h, List<RangesArray> rangesArrayList) {
        //顺便倒置一下顺序
        RangesArray ra = null;
        List<RangesArray> raList = new ArrayList<>();
        //对于宽度过小的切片扩展i + 5
        int moreDistance = i +5;
        int defaultDistance = i;
        for(int index = rangesArrayList.size() - 1;index >= 0;index--){
            if((rangesArrayList.get(index).end - rangesArrayList.get(index).start)<=15){
                i = moreDistance;
            }
            int start = rangesArrayList.get(index).start - i >= 0?rangesArrayList.get(index).start - i:0;
            int end = rangesArrayList.get(index).end + i <= h?rangesArrayList.get(index).end + i:h;
            ra = new RangesArray(start,end);
            raList.add(ra);
            i = defaultDistance;
        }
//        for(RangesArray rangesArray:rangesArrayList){
//            rangesArray.start = rangesArray.start - i >= 0?rangesArray.start - i:0;
//            rangesArray.end = rangesArray.end + i <= h?rangesArray.end + i:h;
//        }
        return raList;
    }

    /**
     * 认为拉升坐上文本区域下上下留白，因为文字识别训练集数据上下有留白
     * 如果文本区域过于宽则认为是少数民族文字的干扰，上下再裁剪
     * @param newPs
     * @return
     */
    private Point[] expandTextArea(Point[] newPs,float ratio,int height) {
        double leftUpY = newPs[0].y;
        double rightUpY = newPs[1].y;
        double leftDownY = newPs[2].y;
        double rightDownY = newPs[3].y;
        double averageHeight = ((rightDownY - rightUpY)+ (leftDownY - leftUpY)) * 0.5;
        if(averageHeight > 30){
            //文本区域过高，则不需要扩大
            ratio = 0.1f;
        }
        double gap = averageHeight * ratio * 0.5;
        newPs[0].y = (leftUpY - gap) > 0?(leftUpY - gap):0;
        newPs[1].y = (rightUpY - gap) > 0?(rightUpY - gap):0;
        newPs[2].y = (leftDownY + gap) <height?(leftDownY + gap):height;
        newPs[3].y = (rightDownY + gap) < height?(rightDownY + gap):height;

        return  newPs;
    }

    public void saveMat(Mat mat,String fileName){
        Bitmap bitmap2 = Bitmap.createBitmap(mat.width(),mat.height(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat,bitmap2);
        ImageUtils.saveBitmap(bitmap2,fileName);
    }

    /**
     * 在膨胀图中寻找文本的区域框
     * @param mat
     */
    public List<RotatedRect> findTextRegion(Mat mat){
        List<MatOfPoint> matOfPointList = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(mat,matOfPointList,hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE);
        List<RotatedRect> regions = new ArrayList<>();
        MatOfPoint2f mat2fsrc = new MatOfPoint2f();
        for(MatOfPoint matOfPoint:matOfPointList){
            double area = Imgproc.contourArea(matOfPoint);
            if(area<500)
                continue;
            matOfPoint.convertTo(mat2fsrc,CvType.CV_32FC2);
            double epsilon = 0.001 * Imgproc.arcLength(mat2fsrc,true);
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(mat2fsrc,approx,epsilon,true);
            RotatedRect rect = Imgproc.minAreaRect(approx);
            int width = rect.boundingRect().width;
            int height = rect.boundingRect().height;
            //System.out.println(width+":"+height+"---------"+rect.size+"-----------"+rect.angle+"-------------"+rect.boundingRect().x+":"+rect.boundingRect().y);
            if(height>width*1.2)
                continue;
            regions.add(rect);
        }
        return regions;
    }


    /**
     * 使用canny、形态学变化提取文本区域
     * @param mat
     * @return
     */
    public Mat extractTextArea(Mat mat){
        Bitmap bitmap1 = Bitmap.createBitmap(mat.width(),mat.height(), Bitmap.Config.ARGB_8888);
        Bitmap bitmap2= Bitmap.createBitmap(mat.width(),mat.height(), Bitmap.Config.ARGB_8888);
        Bitmap bitmap3 = Bitmap.createBitmap(mat.width(),mat.height(), Bitmap.Config.ARGB_8888);
        Bitmap bitmap4 = Bitmap.createBitmap(mat.width(),mat.height(), Bitmap.Config.ARGB_8888);
        Bitmap bitmap5 = Bitmap.createBitmap(mat.width(),mat.height(), Bitmap.Config.ARGB_8888);
        Bitmap bitmap6 = Bitmap.createBitmap(mat.width(),mat.height(), Bitmap.Config.ARGB_8888);
        Point anchor = new Point(-1,-1);
        //2019-11-19新增，先做模糊处理排除噪声干扰 todo 新增判断条件，符合在做高斯模糊
        Mat blur = new Mat();
        Imgproc.blur(mat,blur,new Size(2,2));
        Utils.matToBitmap(blur,bitmap1);
        mainActivity.blur.setImageBitmap(bitmap1);
        //saveMat(blur,"blur.jpg");
        //转成灰度图
        Mat gray = new Mat();
        Imgproc.cvtColor(blur,gray,Imgproc.COLOR_RGB2GRAY);
        Utils.matToBitmap(gray,bitmap2);
        mainActivity.gray.setImageBitmap(bitmap2);
        //saveMat(gray,"gray.jpg");
        //canny算子，求x方向的梯度
        Mat canny = new Mat();
        Imgproc.Canny(gray,canny,60,180);
        Utils.matToBitmap(canny,bitmap3);
        mainActivity.canny.setImageBitmap(bitmap3);
        //saveMat(canny,"canny.jpg");
        //二值化
//        Mat binary = new Mat();
//        Imgproc.threshold(canny,binary,0,255,THRESH_OTSU | THRESH_BINARY);
//        saveMat(binary,"binary.jpg");
        //膨胀
        Mat elementDilate = Imgproc.getStructuringElement(MORPH_RECT,new Size(15,3));  //24,4
        Mat dilate1 = new Mat();
        Imgproc.dilate(canny,dilate1,elementDilate,anchor,1);
        Utils.matToBitmap(dilate1,bitmap4);
        mainActivity.dilate1.setImageBitmap(bitmap4);
        //saveMat(dilate1,"dilate1.jpg");
        //腐蚀
        Mat elementErode = Imgproc.getStructuringElement(MORPH_RECT,new Size(5,10));
        Mat erode1 = new Mat();
        Imgproc.erode(dilate1,erode1,elementErode,anchor,1);
        Utils.matToBitmap(erode1,bitmap5);
        mainActivity.erode.setImageBitmap(bitmap5);
        //saveMat(erode1,"erode1.jpg");
        //再次膨胀,拉长卷积核的长度，让同一行的文字信息可以连接在一起
        elementDilate = Imgproc.getStructuringElement(MORPH_RECT,new Size(40,3));
        Mat dilate2 = new Mat();
        Imgproc.dilate(erode1,dilate2,elementDilate,anchor,1);
        Utils.matToBitmap(dilate2,bitmap6);
        mainActivity.dilate2.setImageBitmap(bitmap6);
        //saveMat(dilate2,"dilate2.jpg");
        return dilate2;
    }

    /**
     * 透视变换
     * @param in 原图像
     * @param point 定位到的四个点
     * @return
     */
    private  Mat getWarpPersPective(Mat in,Point [] point){
        MatOfPoint2f reusltPoint2f=null,srcPoint2f=null;
        Mat out=new Mat();
        Point []targetPoints=new Point[4];
        for (int i=0;i<4;i++){
            targetPoints[i]=new Point();
        }
        //这里拿到倾斜的长度作为宽高 结果可能比真正矫正的图片略小点 但是矫正效果还是很不错的
        double rect_width = Math.sqrt(Math.abs(point[0].x - point[1].x)*Math.abs(point[0].x - point[1].x) +
                Math.abs(point[0].y - point[1].y)*Math.abs(point[0].y - point[1].y));
        double rect_height =  Math.sqrt(Math.abs(point[0].x - point[2].x)*Math.abs(point[0].x - point[2].x) +
                Math.abs(point[0].y - point[2].y)*Math.abs(point[0].y - point[2].y));

        double moveValueX = 0.0;
        double moveValueY = 0.0;

        targetPoints[0].x = 0.0 + moveValueX; targetPoints[0].y = 0 + moveValueY;// top_left
        targetPoints[2].x = 0.0 + moveValueX; targetPoints[2].y = rect_height + moveValueY;// bottom_Left
        targetPoints[1].x = rect_width + moveValueX; targetPoints[1].y = 0.0 + moveValueY;// top_Right
        targetPoints[3].x = rect_width + moveValueX; targetPoints[3].y = rect_height + moveValueY;// bottom_Right
        reusltPoint2f=new MatOfPoint2f(targetPoints);//这里需要将四个点转换成Mat
        srcPoint2f=new MatOfPoint2f(point);

        Mat tranform=Imgproc.getPerspectiveTransform(reusltPoint2f,srcPoint2f); // 透视变换
        Imgproc.warpPerspective(in,out,tranform,new Size(rect_width,rect_height),INTER_LINEAR | WARP_INVERSE_MAP);
        return out;//变换后的Mat
    }

    public void perspectiveChange(float hFactor,float w_factor,List<Coordinate> tensorCoor){
        //点在原图的坐标换算
        List<Coordinate> keyPoint = new ArrayList<>();
        Coordinate c = null;
        for(Coordinate coordinate:tensorCoor){
            c = new Coordinate((int)(coordinate.getX()*w_factor),(int)(coordinate.getY()*hFactor));
            keyPoint.add(c);
        }
    }

    public Mat getWarPerspective(Mat in, Point[] point){
        MatOfPoint2f reusltPoint2f=null,srcPoint2f=null;
        Mat out=new Mat();
        Point []targetPoints=new Point[4];
        for (int i=0;i<4;i++){
            targetPoints[i]=new Point();
        }
        //这里拿到倾斜的长度作为宽高 结果可能比真正矫正的图片略小点 但是矫正效果还是很不错的
        double rect_width = Math.sqrt(Math.abs(point[0].x - point[1].x)*Math.abs(point[0].x - point[1].x) +
                Math.abs(point[0].y - point[1].y)*Math.abs(point[0].y - point[1].y));
        double rect_height =  Math.sqrt(Math.abs(point[0].x - point[2].x)*Math.abs(point[0].x - point[2].x) +
                Math.abs(point[0].y - point[2].y)*Math.abs(point[0].y - point[2].y));

        double moveValueX = 0.0;
        double moveValueY = 0.0;

        targetPoints[0].x = 0.0 + moveValueX; targetPoints[0].y = 0 + moveValueY;// top_left
        targetPoints[2].x = 0.0 + moveValueX; targetPoints[2].y = rect_height + moveValueY;// bottom_Left
        targetPoints[1].x = rect_width + moveValueX; targetPoints[1].y = 0.0 + moveValueY;// top_Right
        targetPoints[3].x = rect_width + moveValueX; targetPoints[3].y = rect_height + moveValueY;// bottom_Right
        reusltPoint2f=new MatOfPoint2f(targetPoints);//这里需要将四个点转换成Mat
        srcPoint2f=new MatOfPoint2f(point);

        Mat tranform= Imgproc.getPerspectiveTransform(reusltPoint2f,srcPoint2f); // 透视变换
        Imgproc.warpPerspective(in,out,tranform,new Size(rect_width,rect_height),INTER_LINEAR | WARP_INVERSE_MAP);
        return out;//变换后的Mat
    }

    /**
     * convert 25*19*4 to 800*600
     * @param maxPointCoordinate
     * @return
     */
    public Point[] convertTensorPoint2OriginImage(List<Float> maxPointCoordinate){
        List<Float> list1 = new ArrayList<>(),list2 = new ArrayList<>() ,list3 = new ArrayList<>(),list4 = new ArrayList<>();
        for(int i = 1;i<=maxPointCoordinate.size();i++){
            switch (i%4){
                case 1:
                    list1.add(maxPointCoordinate.get(i-1));
                    break;
                case 2:
                    list2.add(maxPointCoordinate.get(i-1));
                    break;
                case 3:
                    list3.add(maxPointCoordinate.get(i-1));
                    break;
                case 0:
                    list4.add(maxPointCoordinate.get(i-1));
                    break;
                default:
                    break;
            }
        }
        int firstIndex = list1.indexOf(Collections.max(list1));
        int secondIndex = list2.indexOf(Collections.max(list2));
        int thirdIndex = list3.indexOf(Collections.max(list3));
        int fourthIndex = list4.indexOf(Collections.max(list4));
        List<Integer> indexArray = new ArrayList<>(Arrays.asList(firstIndex,secondIndex,thirdIndex,fourthIndex));
        Point[] points = new Point[4];
        Point point = null;
        int i = 0;
        float bx = 800f/25;
        float by = 600f/19;
        for(Integer index:indexArray){
            int ys = 0;
            if((ys = ((index+1) % 25)) == 0){
                ys = 25;
            }
            int x = (int) (ys * bx );
            int y = (int) ((index+1) * 1f /25 * by);
            point = new Point();
            point.x = x;
            point.y = y;
            points[i] = point;
            i++;
        }
        return points;
    }

    public Point[] dsnt(List<Float> heatmaps){
        // norm the heatmaps
        int x = 0;
        int y = 0;
        int dim = 4;
        float[] max_axis = {0,0,0,0};
        float[] norm = {0,0,0,0};
        List<Float> list1 = new ArrayList<>(),list2 = new ArrayList<>() ,list3 = new ArrayList<>(),list4 = new ArrayList<>();
        for(int i = 1;i<=heatmaps.size();i++){
            switch (i%4){
                case 1:
                    list1.add(heatmaps.get(i-1));
                    break;
                case 2:
                    list2.add(heatmaps.get(i-1));
                    break;
                case 3:
                    list3.add(heatmaps.get(i-1));
                    break;
                case 0:
                    list4.add(heatmaps.get(i-1));
                    break;
                default:
                    break;
            }
        }
        //把模型返回的float[]转换为一个list4 * (25*19)
        List<List<Float>> listList = new ArrayList<>(Arrays.asList(list1,list2,list3,list4));
        //遍历4
        for(int i=0;i<dim;i++){
            //寻找最大值
            max_axis[i] = Collections.max(listList.get(i));
            for(int j=0;j<listList.get(i).size();j++){
                //对每一个元素与最大值的差做exp操作
                listList.get(i).set(j, (float) Math.exp(listList.get(i).get(j) - max_axis[i]));
                norm[i] += listList.get(i).get(j);
            }
        }
        for(int i=0;i<dim;i++){
            for(int j=0;j<listList.get(i).size();j++){
                listList.get(i).set(j, listList.get(i).get(j)*1.0f/norm[i]);
            }
        }
//        float output_x = 2 - 24f/25;
//        float output_y = 2 - 18f/19;
        Float[] output_x = new Float[]{-0.96f ,-0.88f ,-0.8f , -0.72f ,-0.64f ,-0.56f ,-0.48f ,-0.4f  ,-0.32f ,-0.24f ,-0.16f ,-0.08f , 0.f   , 0.08f , 0.16f , 0.24f , 0.32f , 0.4f ,  0.48f , 0.56f , 0.64f , 0.72f ,0.8f, 0.88f, 0.96f};
        Float[] output_y = new Float[]{-0.94736842f, -0.84210526f, -0.73684211f, -0.63157895f, -0.52631579f,
                -0.42105263f, -0.31578947f, -0.21052632f, -0.10526316f , 0.f,
                0.10526316f,  0.21052632f , 0.31578947f , 0.42105263f,  0.52631579f,
                0.63157895f,  0.73684211f, 0.84210526f , 0.94736842f};
        Point[] pointList = new Point[4];
        Point point = null;
        List<Float> dsnt_x = new ArrayList<>();
        List<Float> dsnt_y = new ArrayList<>();
        for(int i = 0;i<dim;i++){
            float sum_x = 0f;
            float sum_y = 0f;
            for (int j=0;j<listList.get(i).size();j++){
                int selectX = 0;
                int selectY = 0;

                selectX = j%25;
                selectY = j / 25;
                sum_x += listList.get(i).get(j)*output_x[selectX];
                sum_y += listList.get(i).get(j)*output_y[selectY];
            }
            dsnt_x.add(sum_x);
            dsnt_y.add(sum_y);
        }
        //reduce sum
        for (int i=0;i<dim;i++){
            x = (int)((dsnt_x.get(i) +1 )/2*800);
            y = (int)((dsnt_y.get(i) +1)/2*600);
            point = new Point(x,y);
            pointList[i] = point;
        }
        return pointList;
    }

    class Coordinate{
        int x;
        int y;

        public int getX() {
            return x;
        }

        public void setX(int x) {
            this.x = x;
        }

        public int getY() {
            return y;
        }

        public void setY(int y) {
            this.y = y;
        }

        public Coordinate(int x, int y) {
            this.x = x;
            this.y = y;
        }

        @Override
        public String toString() {
            return "Coordinate{" +
                    "x=" + x +
                    ", y=" + y +
                    '}';
        }
    }

    public void close(){
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
        tfliteModel = null;
    }

    /** Loads input image, and applies preprocessing. */
    private TensorImage loadImage(final Bitmap bitmap, int sensorOrientation) {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap);

        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        int numRoration = sensorOrientation / 90;
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(600, 800, ResizeOp.ResizeMethod.BILINEAR))
                        .build();
        return imageProcessor.process(inputImageBuffer);
    }

    /**
     * 内部类
     * 保存切片的开始和结束位置
     */
    public class RangesArray {
        private int start;
        private int end;

        public int getStart() {
            return start;
        }

        public void setStart(int start) {
            this.start = start;
        }

        public int getEnd() {
            return end;
        }

        public void setEnd(int end) {
            this.end = end;
        }

        public RangesArray(int start, int end) {
            this.start = start;
            this.end = end;
        }

        @Override
        public String toString() {
            return "RangesArray{" +
                    "start=" + start +
                    ", end=" + end +
                    '}';
        }
    }
}

