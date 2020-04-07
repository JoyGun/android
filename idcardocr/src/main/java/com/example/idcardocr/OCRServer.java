package com.example.idcardocr;

import android.app.Activity;
import android.app.ActivityManager;
import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class OCRServer {
    private static PointDetector pointDetector = null;
    private static ChineseCharacterRecognize chineseCharacterRecognize = null;
    private static final String SIGNATURE = "com.sunyard.gfrcu";
    private Activity context;

    public static String NATIONS = "汉|满|蒙古|回|藏|维吾尔|苗|彝|壮|布依|侗|瑶|白|土家|哈尼|哈萨克|傣|黎|傈僳|佤|畲|高山|拉祜|水|东乡|纳西|景颇|柯尔克孜|土|达斡尔|仫佬|羌|布朗|撒拉|毛南|仡佬|锡伯|阿昌|普米|朝鲜|塔吉克|怒|乌孜别克|俄罗斯|鄂温克|德昂|保安|裕固|京|塔塔尔|独龙|鄂伦春|赫哲|门巴|珞巴|基诺";
    public static String CITIES = "北京|天津|上海|重庆|河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|河南|湖北|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|台湾|内蒙古|广西|西藏|宁夏|新疆|香港|澳门|广州|沈阳|南京|武汉|成都|西安|大连|长春|哈尔滨|济南|青岛|杭州|宁波|厦门|深圳|石家庄|太原|沈阳|合肥|福州|南昌|济南|郑州|长沙|海口|贵阳|昆明|兰州|西宁|台北|呼和浩特|南宁|拉萨|银川|乌鲁木齐";
    /**
     * @赖林晓
     * 小知识：末尾的"肖"为后来单独添加，二次简化字后"萧"统一改为"肖",
     * 所以本来并无肖姓，后来改回来了但是由于户口改动太大，
     * 所以姓氏"肖"并为改回来。查了好几个常见姓氏，里面都无"肖"**/
    public static String FIRSTNAME = "辗迟|荔菲|段干|公良|闾丘|锺离|鲜于|亓官|司寇|子车|颛孙|端木|巫马|公西|漆雕|乐正|壤驷|拓跋|夹谷|宰父|谷梁|百里|东郭|南门|呼延|羊舌|微生|梁丘|左丘|东门|西门|南宫|第五|凃肖|司空|司徒|慕容|长孙|宇文|钟离|令狐|轩辕|仲孙|公孙|申屠|太叔|单于|淳于|濮阳|宗政|公冶|澹台|公羊|尉迟|皇甫|赫连|东方|闻人|诸葛|夏侯|欧阳|上官|司马|万俟|相查|空曾|聂晁|东欧|步都|第五|南宫|西门|东门|左丘|梁丘|微生|羊舌|呼延|南门|东郭|百里|段干|谷梁|宰父|夹谷|拓跋|公良|壤驷|乐正|漆雕|公西|巫马|端木|颛孙|子车|司寇|亓官|司空|司徒|闾丘|鲜于|慕容|长孙|宇文|钟离|令狐|轩辕|仲孙|公孙|申屠|太叔|单于|淳于|濮阳|宗政|公冶|澹台|公羊|尉迟|皇甫|赫连|东方|闻人|诸葛|夏侯|欧阳|上官|司马|万俟|赵|钱|孙|李|周|吴|郑|王|冯|陈|褚|卫|蒋|沈|韩|杨|朱|秦|尤|许|何|吕|施|张|孔|曹|严|华|金|魏|陶|姜|戚|谢|邹|喻|柏|水|窦|章|云|苏|潘|葛|奚|范|彭|郎|鲁|韦|昌|马|苗|凤|花|方|俞|任|袁|柳|酆|鲍|史|唐|费|廉|岑|薛|雷|贺|倪|汤|滕|殷|罗|毕|郝|邬|安|常|乐|于|时|傅|皮|卞|齐|康|伍|余|元|卜|顾|孟|平|黄|和|穆|萧|尹|姚|邵|湛|汪|祁|毛|禹|狄|米|贝|明|臧|计|伏|成|戴|谈|宋|茅|庞|熊|纪|舒|屈|项|祝|董|梁|杜|阮|蓝|闵|席|季|麻|强|贾|路|娄|危|江|童|颜|郭|梅|盛|林|刁|钟|徐|邱|骆|高|夏|蔡|田|樊|胡|凌|霍|虞|万|支|柯|昝|管|卢|莫|经|房|裘|缪|干|解|应|宗|丁|宣|贲|邓|郁|单|杭|洪|包|诸|左|石|崔|吉|钮|龚|程|嵇|邢|滑|裴|陆|荣|翁|荀|羊|於|惠|甄|麴|家|封|芮|羿|储|靳|汲|邴|糜|松|井|段|富|巫|乌|焦|巴|弓|牧|隗|山|谷|车|侯|宓|蓬|全|郗|班|仰|秋|仲|伊|宫|宁|仇|栾|暴|甘|钭|厉|戎|祖|武|符|刘|景|詹|束|龙|叶|幸|司|韶|郜|黎|蓟|薄|印|宿|白|怀|蒲|邰|从|鄂|索|咸|籍|赖|卓|蔺|屠|蒙|池|乔|阴|欎|胥|能|苍|双|闻|莘|党|翟|谭|贡|劳|逄|姬|申|扶|堵|冉|宰|郦|雍|舄|璩|桑|桂|濮|牛|寿|通|边|扈|燕|冀|郏|浦|尚|农|温|别|庄|晏|柴|瞿|阎|充|慕|连|茹|习|宦|艾|鱼|容|向|古|易|慎|戈|廖|庾|终|暨|居|衡|步|都|耿|满|弘|匡|国|文|寇|广|禄|阙|东|殴|殳|沃|利|蔚|越|夔|隆|师|巩|厍|聂|晁|勾|敖|融|冷|訾|辛|阚|那|简|饶|空|曾|毋|沙|乜|养|鞠|须|丰|巢|关|蒯|相|查|後|荆|红|游|竺|权|逯|盖|益|桓|公|仉|督|晋|楚|闫|法|汝|鄢|涂|钦|归|海|岳|帅|缑|亢|况|后|有|琴|商|牟|佘|佴|伯|赏|墨|哈|谯|笪|年|爱|阳|佟|言|福|百|家|姓|终|寸|卓|蔺|屠|蒙|池|乔|阳|郁|胥|能|苍|双|闻|莘|党|翟|谭|贡|劳|逄|姬|申|扶|堵|冉|宰|郦|雍|却|璩|桑|桂|濮|牛|寿|通|边|扈|燕|冀|僪|浦|尚|农|温|别|庄|晏|柴|瞿|阎|充|慕|连|茹|习|宦|艾|鱼|容|向|古|易|慎|戈|庾|终|暨|居|衡|耿|满|弘|匡|国|文|寇|广|禄|阙|殳|沃|利|蔚|越|夔|隆|师|巩|厍|勾|敖|融|冷|訾|辛|阚|那|简|饶|毋|沙|乜|养|鞠|须|丰|巢|关|蒯|后|荆|红|游|竺|权|逮|盍|益|桓|公|唱|召|有|舜|丛|岳|寸|贰|皇|侨|彤|竭|端|赫|实|甫|集|象|翠|狂|辟|典|良|函|芒|苦|其|京|中|夕|之|蹇|称|诺|来|多|繁|戊|朴|回|毓|税|荤|靖|绪|愈|硕|牢|买|但|巧|枚|撒|泰|秘|亥|绍|以|壬|森|斋|释|奕|姒|朋|求|羽|用|占|真|穰|翦|闾|漆|贵|代|贯|旁|崇|栋|告|休|褒|谏|锐|皋|闳|在|歧|禾|示|是|委|钊|频|嬴|呼|大|威|昂|律|冒|保|系|抄|定|化|莱|校|么|抗|祢|綦|悟|宏|功|庚|务|敏|捷|拱|兆|丑|丙|畅|苟|随|类|卯|俟|友|答|乙|允|甲|留|尾|佼|玄|乘|裔|延|植|环|矫|赛|昔|侍|度|旷|遇|偶|前|由|咎|塞|敛|受|泷|袭|衅|叔|圣|御|夫|仆|镇|藩|邸|府|掌|首|员|焉|戏|可|智|尔|凭|悉|进|笃|厚|仁|业|肇|资|合|仍|九|衷|哀|刑|俎|仵|圭|夷|徭|蛮|汗|孛|乾|帖|罕|洛|淦|洋|邶|郸|郯|邗|邛|剑|虢|隋|蒿|茆|菅|苌|树|桐|锁|钟|机|盘|铎|斛|玉|线|针|箕|庹|绳|磨|蒉|瓮|弭|刀|疏|牵|浑|恽|势|世|仝|同|蚁|止|戢|睢|冼|种|己|泣|潜|卷|脱|谬|蹉|赧|浮|顿|说|次|错|念|夙|斯|完|丹|表|聊|源|姓|吾|寻|展|出|不|户|闭|才|无|书|学|愚|本|性|雪|霜|烟|寒|少|字|桥|板|斐|独|千|诗|嘉|扬|善|揭|祈|析|赤|紫|青|柔|刚|奇|拜|佛|陀|弥|阿|素|长|僧|隐|仙|隽|宇|祭|酒|淡|塔|琦|闪|始|星|南|天|接|波|碧|速|禚|腾|潮|镜|似|澄|潭|謇|纵|渠|奈|风|春|濯|沐|茂|英|兰|檀|藤|枝|检|生|折|登|驹|骑|貊|虎|肥|鹿|雀|野|禽|飞|节|宜|鲜|粟|栗|豆|帛|官|布|衣|藏|宝|钞|银|门|盈|庆|喜|及|普|建|营|巨|望|希|道|载|声|漫|犁|力|贸|勤|革|改|兴|亓|睦|修|信|闽|北|守|坚|勇|汉|练|尉|士|旅|五|令|将|旗|军|行|奉|敬|恭|仪|母|堂|丘|义|礼|慈|孝|理|伦|卿|问|永|辉|位|让|尧|依|犹|介|承|市|所|苑|杞|剧|第|零|谌|招|续|达|忻|六|鄞|战|迟|候|宛|励|粘|萨|邝|覃|辜|初|楼|城|区|局|台|原|考|妫|纳|泉|老|清|德|卑|过|麦|曲|竹|百|福|言|佟|爱|年|笪|谯|哈|墨|赏|伯|佴|佘|牟|商|琴|后|况|亢|缑|帅|海|归|钦|鄢|汝|法|闫|楚|晋|督|仉|盖|逯|库|郏|逢|阴|薄|厉|稽|开|光|操|瑞|眭|泥|运|摩|伟|铁|迮|肖";
    public static List<String> ID_END = new ArrayList(Arrays.asList("0","1","2","3","4","5","6","7","8","9","X"));

    private static OCRServer ocrServer = null;

    private OCRServer(Activity activity){
        this.context = activity;
    }

    public static OCRServer createServer(Activity activity){
        if(ocrServer == null){
            ocrServer = new OCRServer(activity);
        }
        return ocrServer;
    }

    /**
     * 先切片然后识别
     *
     * @param mBitmap 需要识别的原始图像
     * @return
     */
    public Map regImg(Bitmap mBitmap, int regSide) {
        if(!initModel()){
            return (Map) new HashMap().put("permission","您无调用权限");
        }
        String line = null;
        Map result = new HashMap();
        Map detectTextAreaResult = pointDetector.detectImage(mBitmap,regSide);
        List<Bitmap> bitmapList = (List<Bitmap>) detectTextAreaResult.get("textArea");
        Boolean isRect = (Boolean) detectTextAreaResult.get("isRect");
        List<String> textLineByLine = new ArrayList<>();
        for (Bitmap bitmap : bitmapList) {
            line = chineseCharacterRecognize.recognizeChineseImage(bitmap);
            if (!line.replace(" ", "").equals("")) {
                textLineByLine.add(line);
            }
        }
        if(regSide == IdCardRegConstant.FRONT_SIDE){
            closeModel();
            return extractInfo(textLineByLine,isRect);
        }else{
            closeModel();
            return extractBackInfo(textLineByLine,isRect);
        }
    }

    /**
     * 正面结果抽取
     * @param textLineByLine 识别出来的切片信息
     * @param isRect
     * @return
     */
    public static Map extractInfo(List<String> textLineByLine,Boolean isRect) {
        Map<String, String> result = new HashMap<String, String>() {{
            put("性别", "");
            put("姓名", "");
            put("民族", "");
            put("出生年月", "");
            put("住址", "");
            put("公民身份证号码", "");
        }};
        String word = null;
        String line = null;

        for (int max = textLineByLine.size() - 1; max >= 0; max--) {
            line = textLineByLine.get(max);
            System.out.println(line);
            //当找到性别/民族时不再寻找姓名，因为地址字段汉字多很容易错误匹配
            if (!findAll(line, "[姓名]").equals("") && result.get("姓名").equals("")&&result.get("性别").equals("")&&result.get("民族").equals("")) {
                //此逻辑必须找到"名"
                word = getTextMatch(line, "(?:姓.名|姓.|名)([\\u4e00-\\u9fa5]+)");
                if (!word.equals("")) {
                    //检测首字是否落入百家姓，不是则表明只识别出"姓"，先用姓名长度排除少数民族
                    if(word.length()<5&&word.length()>1){
                        if(findAll(String.valueOf(word.charAt(0)),FIRSTNAME).equals("")&&findAll(word.substring(0,2),FIRSTNAME).length()!=2){
                            word = getTextMatch(line, "(?:姓)([\\u4e00-\\u9fa5]+)");
                        }
                    }
                    result.put("姓名", word);
                }
            } else if (!findAll(line, "[性别民族男女汉]").equals("") && result.get("民族").equals("") && result.get("性别").equals("")) {
                word = getTextMatch(line, "(男|女)");
                if (!word.equals("")) {
                    result.put("性别", word);
                } else {
                    //result.put("性别", "男");
                }
                word = null;
                word = findAll(line, NATIONS);
                if (!word.equals("")) {
                    result.put("民族", word);
                } else {
                    //result.put("民族", "汉");
                }
            } else if (!findAll(line, "[出生年月日]").equals("") && result.get("出生年月").equals("") && findAll(line, "\\d{10,18}").equals("")) {
                Matcher m = Pattern.compile("([12]\\d+)\\D*(\\d+)\\D*(\\d+)\\D*").matcher(line);
                if (m.find()) {
                    String year = m.group(1);
                    String month = m.group(2);
                    String day = m.group(3);
                    month = standardBirthData(month);
                    day = standardBirthData(day);
                    result.put("出生年月", year + month + day);
                }
            } else if ((!findAll(line, "(住.?|.?址)(.*)").equals("") || !findAll(line, CITIES).equals("")) && result.get("住址").equals("")) {
                //寻找地址，由于已经用定点检测框定了卡范围，剩余的在身份证号码之前的都认为是地址字段
                Matcher m = Pattern.compile("(住.?|.?址)(.*)").matcher(line);
                String city = findAll(line, CITIES);
                if (!city.equals("")) {
                    word = getTextMatch(line, "((?:"+city+")\\w+)");
                    if (!word.equals("")) {
                        result.put("住址", word);
                    }
                } else if (m.find()) {
                    int idx = line.indexOf("址");
                    if (idx >= 0 && idx <= 2) {
                        result.put("住址", line.substring(idx, line.length()));
                    }
                }
                if (m.find() || !city.equals("")) {
                    //向下一行寻找地址信息
                    for (int temp = max -1; temp > 0; temp--) {
                        if(temp<0){
                            break;
                        }
                        line = textLineByLine.get(temp);
                        if (!findAll(line, "公民|身份|号码").equals("") || !findAll(line, "\\d{10,18}").equals("")) {
                            word = findAll(line, "\\d{10,}");
                            word = extractIDNumber(word);
                            result.put("公民身份证号码", word);
                            if(IdCardUtil.isValidatedAllIdcard(word) && Integer.parseInt(word.substring(word.length() - 12, word.length() - 8)) > 1900 && Integer.parseInt(word.substring(word.length() - 12, word.length() - 8))< (Calendar.getInstance().get(Calendar.YEAR)-15)){
                                //因为有的时候错误的也能通过校验位，所以加一个限定需要生日在1900年以后且在当前年份的15年之前
                                break;
                            }
                        } else {
                            result.put("住址", result.get("住址") + line);
                        }
                    }
                }
            } else {
                if (!findAll(line, "公民|身份|号码").equals("") || !findAll(line, "\\d{10,18}").equals("")) {
                    word = findAll(line, "\\d{10,}");
                    word = extractIDNumber(word);
                    result.put("公民身份证号码", word);
                    if(IdCardUtil.isValidatedAllIdcard(word)&& Integer.parseInt(word.substring(word.length() - 12, word.length() - 8)) > 1900 && Integer.parseInt(word.substring(word.length() - 12, word.length() - 8))< (Calendar.getInstance().get(Calendar.YEAR)-15)){
                        break;
                    }else {
                        //todo 补充一些修正号码的逻辑
                    }
                }
            }
        }
        //如果姓名为空则找倒数汉字行
        if (result.get("姓名").equals("")) {
            for (int max = textLineByLine.size() - 1; max >= 0; max--) {
                //从最后一行开始找，如果有字出现在百家姓中则认为是姓名
                line = textLineByLine.get(max);
                //剔除全部非汉字
                //line = line.replaceAll("[^\u4e00-\u9fa5]","");
                if (!findAll(line, FIRSTNAME).equals("")&&line.length()>=2) {
                    if(line.length()>=5){
                        //认为如果text长度大于4则是“姓名”两个字都识别出错了，先排除姓名两个字
                        line = line.substring(2,line.length());
                    }
//                    if (line.length() < 4) {
//                        result.put("姓名", line);
//                    } else if (line.length() == 4) {
//                        result.put("姓名", line.substring(2, 4));
//                    } else {
//                        result.put("姓名", line.substring(line.length() - 3, line.length()));
//                    }
                    if(!findAll(line, FIRSTNAME).equals("")){
                        result.put("姓名", line.substring(line.indexOf(findAll(line,FIRSTNAME)),line.length()));
                    }
                    break;
                }
            }
        }
        //根据身份证号码修正
        if (IdCardUtil.isValidatedAllIdcard(result.get("公民身份证号码"))) {
            String idNumber = result.get("公民身份证号码");
            int sex_vali = idNumber.charAt(idNumber.length() - 2);
            if (sex_vali % 2 == 1) {
                result.put("性别", "男");
            } else {
                result.put("性别", "女");
            }
            result.put("出生年月", idNumber.substring(idNumber.length() - 12, idNumber.length() - 4));
        }

        return result;
    }

    /**
     * 背面的识别结果
     * @param textLineByLine 识别出来的切片信息
     * @param isRect
     * @return
     */
    public static Map extractBackInfo(List<String> textLineByLine,Boolean isRect){
        Map<String, String> result = new HashMap<String, String>() {{
            put("签发机关","");
            put("有效期限","");
        }};
        String word = null;
        //String line = null;
        for(String line:textLineByLine){
            System.out.println(line);
            if(result.get("签发机关").equals("")){
                word = getTextMatch(line,"机关(\\w+)");
                if(word!=null &&word.equals("")){
                    word = getTextMatch(line,"发机[头类关美]?(\\w+)");
                    if(word!=null &&word.equals("")){
                        word = getTextMatch(line,"机美(\\w+)");
                    }
                }
                if(word!=null && word.length()>0){
                    if(String.valueOf(word.charAt(0)).equals("一")){
                        //有时会多识别出一个"一"
                        word = word.substring(1,word.length());
                    }
                    result.put("签发机关",word);
                }else{
                    //找后面的公安分局字样
                    word = findAll(line,"分局|公安局");
                    if(word!=null &&word.length()>4){
                        if(String.valueOf(word.charAt(0)).equals("一")){
                            //有时会多识别出一个"一"
                            word = word.substring(1,word.length());
                        }
                        word = word.substring(4,word.length());
                        result.put("签发机关",word);
                    }
                }
            }
            if(result.get("有效期限").equals("")){
                line = line.replaceAll("Q","0").replaceAll("B","8");
                //获取全部的数字
                word = getTextMatch(line.replaceAll("[^0-9]",""),"(\\d{8,})");
                //System.out.println("word"+word);
                if(word!=null &&!word.equals("")){
                    if(word.length()==8){
                        result.put("有效期限",word+"至长期");
                    }else if(word.length()==16){
                        result.put("有效期限",word.substring(0,8)+"至"+word.substring(8,16));
                    }else {
                        if(word.length()>8)
                            result.put("有效期限",word.substring(0,8)+"至"+word.substring(8,word.length()));
                    }
                }
            }
        }
        return result;
    }

    public static String getTextMatch(String text, String pattern) {
        Matcher m = Pattern.compile(pattern).matcher(text.replace(" ", ""));
        if (m.find()) {
            return m.group(1);
        } else {
            return "";
        }
    }

    public static String findAll(String text, String pattern) {
        Matcher m = Pattern.compile(pattern).matcher(text.replace(" ", ""));
        if (m.find()) {
            return m.group(0);
        } else {
            return "";
        }
    }

    /**
     * 提取号码的逻辑用到了两次，单独分装一下
     * @param word 识别出来的号码字段
     * @return 解析好后的号码
     */
    public static String extractIDNumber(String word){
        if (word.length() == 18) {
            //result.put("公民身份证号码", word);
        } else if (word.length() == 17) {
            //补足最后一位，使之满足校验
            for(String end:ID_END){
                if(IdCardUtil.isValidatedAllIdcard(word+end)){
                    word = word + end;
                    break;
                }
            }
        } else if (word.length() == 19) {
            //挨个删除做校验
            for(int surplus = 0;surplus<19;surplus++){
                if(IdCardUtil.isValidatedAllIdcard(word.substring(0,surplus)+word.substring(surplus+1,19))){
                    word = word.substring(0,surplus)+word.substring(surplus+1,19);
                    break;
                }
            }
        }
        return word;
    }

    /**
     * 为月日前面补充0
     *
     * @param data
     * @return
     */
    private static String standardBirthData(String data) {
        if (data.length() == 1) {
            data = "0" + data;
        }
        return data;
    }

//    String invokerPkg = getAppPkg(Binder.getCallingPid());

    public Boolean initModel() {
        try{
            if(isSignature()){
                pointDetector = PointDetector.create(context,null,null,1);
                chineseCharacterRecognize = ChineseCharacterRecognize.create(context,null,null,1);
                //init opencv
                if (!OpenCVLoader.initDebug()) {
                    OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, context, mLoaderCallback);
                } else {
                    mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
                }
                return true;
            }else {
                Log.e("ocr server","无调用权限");
                return false;
            }

        }catch (Exception e){
            e.printStackTrace();
        }
        return false;
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(context) {
        @Override
        public void onManagerConnected(int status) {
            super.onManagerConnected(status);
            switch (status) {
                case BaseLoaderCallback.SUCCESS:
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    public static void closeModel(){
        pointDetector.close();
        chineseCharacterRecognize.close();
    }

    /**
     * 限制上层的调用名
     * @return 是否合法的调用
     */
    private boolean isSignature() {
        String appId = "";
        try{
//            Class activityClass = Class.forName("android.app.Activity");
//            Field field = activityClass.getDeclaredField("mReferrer");
//            field.setAccessible(true);
//            appId = (String) field.get(context);
            appId = getCurrentPkgName(context);
            Log.i("ocr server appid",appId);
        }catch (Exception e){
            e.printStackTrace();
        }
        //return appId.equals(SIGNATURE);
        return true;
    }

    /**
     * 注意： getRunningAppProcesses()方法在5.0开始，就只返回自身应用的进程，所以只能判断自身进程状态，
     * 如果是400，返回为null,不能拿到当前栈顶Activity的包名
     *
     * @param context
     * @return
     */
    private static String getCurrentPkgName(Context context) {
        // 5x系统以后利用反射获取当前栈顶activity的包名.
        ActivityManager.RunningAppProcessInfo currentInfo = null;
        Field field = null;
        int startTaskToFront = 2;
        String pkgName = null;
        try {
            // 通过反射获取进程状态字段.
            field = ActivityManager.RunningAppProcessInfo.class.getDeclaredField("processState");
        } catch (Exception e) {
            e.printStackTrace();
        }
        ActivityManager am = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
        List appList = am.getRunningAppProcesses();
        ActivityManager.RunningAppProcessInfo app;
        for (int i = 0; i < appList.size(); i++) {
            //ActivityManager.RunningAppProcessInfo app : appList
            app = (ActivityManager.RunningAppProcessInfo) appList.get(i);
            //表示前台运行进程.
            if (app.importance == ActivityManager.RunningAppProcessInfo.IMPORTANCE_FOREGROUND) {
                Integer state = null;
                try {
                    // 反射调用字段值的方法,获取该进程的状态.
                    state = field.getInt(app);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                // 根据这个判断条件从前台中获取当前切换的进程对象
                if (state != null && state == startTaskToFront) {
                    currentInfo = app;
                    break;
                }
            }
        }
        if (currentInfo != null) {
            pkgName = currentInfo.processName;
        }
        return pkgName;
    }
}
