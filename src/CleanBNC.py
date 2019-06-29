import os
import re

ROOT_DIR = "./Corpora/BNC XML/2554/download/Texts"
OUT_SUF  = "_CleanXML"

ALPH_NUM   = "a-zA-Z0-9"
ACCNT_TKNS = "À-ÿ"
PUNC       = "\…\.\!\,\:\;\_\-\–\—\?\'\‘\’\′\″\""
CLS_TKNS   = "\(\)\[\]\{\}"
MATH_TKNS  = "=\+\*\%"
GREEK_ALPH = "ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω"
CURR_TKNS  = "$£¥₩₽₾₺₴₹฿"
SPCL_TKNS  = "\&\•\/\\@©™°"
MATH_SPCL  = "½⅓⅔¼‌¾‌⅕⅖⅗⅘⅙⅚⅐⅛⅜⅝⅞⅑⅒⁰ⁱ¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ"

TKN_CHARS = " " + ALPH_NUM + ACCNT_TKNS + PUNC + CLS_TKNS + MATH_TKNS + GREEK_ALPH + CURR_TKNS + SPCL_TKNS + MATH_SPCL
XML_CHARS = "<>"

VAL_CHARS    = re.compile("[^" + TOK_CHARS + XML_CHARS + "]")
VAL_CHARS_re = "_"

tag_1     = re.compile("<corr[" + TOK_CHARS + "]+>")
tag_1_re  = ""

tag_2     = re.compile("</corr>")
tag_2_re  = ""

tag_3     = re.compile("<p[a-zA-Z\"= ]*>")
tag_3_re  = ""

tag_4     = re.compile("</p>")
tag_4_re  = ""

tag_5     = re.compile("<hi[a-zA-Z\"= ]+>")
tag_5_re  = ""

tag_6     = re.compile("</hi>")
tag_6_re  = ""

tag_7     = re.compile("<item[" + TOK_CHARS + "]*>")
tag_7_re  = ""

tag_8     = re.compile("</item>")
tag_8_re  = ""

tag_9     = re.compile("<label[a-zA-Z\"= ]*>")
tag_9_re  = ""

tag_10     = re.compile("</label>")
tag_10_re  = ""

tag_11    = re.compile("<div[" + TOK_CHARS + "]*>")
tag_11_re = ""

tag_12    = re.compile("</div>")
tag_12_re = ""

tag_13    = re.compile("<list>")
tag_13_re = ""

tag_14    = re.compile("</list>")
tag_14_re = ""

tag_15    = re.compile("</wtext></bncDoc>")
tag_15_re = ""

tag_16    = re.compile("<pb[" + TOK_CHARS + "]*>")
tag_16_re = ""

tag_17    = re.compile("<gap[" + TOK_CHARS + "]*>")
tag_17_re = ""

tag_18    = re.compile("</gap>")
tag_18_re = ""

tag_19    = re.compile("<note[" + TOK_CHARS + "]*>")
tag_19_re = ""

tag_20    = re.compile("</note>")
tag_20_re = ""

tag_21    = re.compile("<quote[" + TOK_CHARS + "]*>")
tag_21_re = ""

tag_22    = re.compile("</quote>")
tag_22_re = ""

tag_23    = re.compile("<stage[" + TOK_CHARS + "]*>")
tag_23_re = ""

tag_24    = re.compile("</stage>")
tag_24_re = ""

tag_25    = re.compile("<head[" + TOK_CHARS + "]*>")
tag_25_re = ""

tag_26    = re.compile("</head>")
tag_26_re = ""

tag_27    = re.compile("<l[" + TOK_CHARS + "]*>")
tag_27_re = ""

tag_28    = re.compile("</l>")
tag_28_re = ""

tag_29    = re.compile("<lg[" + TOK_CHARS + "]*>")
tag_29_re = ""

tag_30    = re.compile("</lg>")
tag_30_re = ""

tag_31    = re.compile("<sp[" + TOK_CHARS + "]*>")
tag_31_re = ""

tag_32    = re.compile("</sp>")
tag_32_re = ""

tag_33    = re.compile("<speaker[" + TOK_CHARS + "]*>")
tag_33_re = ""

tag_34    = re.compile("</speaker>")
tag_34_re = ""

tag_35    = re.compile("<bibl[" + TOK_CHARS + "]*>")
tag_35_re = ""

tag_36    = re.compile("</bibl>")
tag_36_re = ""

tag_37    = re.compile("<unclear[" + TOK_CHARS + "]*>")
tag_37_re = ""

tag_38    = re.compile("</unclear>")
tag_38_re = ""

tag_39    = re.compile("<trunc[" + TOK_CHARS + "]*>")
tag_39_re = ""

tag_40    = re.compile("</trunc>")
tag_40_re = ""

tag_41    = re.compile("<pause[" + TOK_CHARS + "]*>")
tag_41_re = ""

tag_42    = re.compile("</pause>")
tag_42_re = ""

tag_43    = re.compile("<event[" + TOK_CHARS + "]*>")
tag_43_re = ""

tag_44    = re.compile("</event>")
tag_44_re = ""

tag_45    = re.compile("<u[" + TOK_CHARS + "]*>")
tag_45_re = ""

tag_46    = re.compile("</u>")
tag_46_re = ""

tag_47    = re.compile("<vocal[" + TOK_CHARS + "]*>")
tag_47_re = ""

tag_48    = re.compile("</vocal>")
tag_48_re = ""

tag_49    = re.compile("<align[" + TOK_CHARS + "]*>")
tag_49_re = ""

tag_50    = re.compile("<shift[" + TOK_CHARS + "]*>")
tag_50_re = ""

tag_CM_0    = re.compile("<!--.*-->")
tag_CM_0_re = ""

tag_CM_1    = re.compile("<!--.*")
tag_CM_1_re = ""

tag_CM_2    = re.compile(".*-->")
tag_CM_2_re = ""

pat_1    = re.compile("<mw c5=\"[A-Z0-9]+\">")
pat_1_re = ""

pat_2    = re.compile("</mw>")
pat_2_re = ""

pat_3    = re.compile("pos=\"[A-Z]+\"")
pat_3_re = ""

pat_4    = re.compile("hw=")
pat_4_re = "lemma="

rep_1    = re.compile("^((</w>)?(</s>)?)+")
rep_1_re = ""

rep_2    = re.compile(" >[" + TOK_CHARS + "]+$")
rep_2_re = ">"

rep_3    = re.compile("/>$")
rep_3_re = "/></s>"

rep_4    = re.compile("</s>.+$")
rep_4_re = "</s>"

CORR_SENT = re.compile("^<s n=\"[0-9_]+\">[^" + TOK_CHARS + XML_CHARS + "]+</s>$")
EMPTY     = re.compile("^$")


def CleanSent(sent):
    sent = VAL_CHARS.sub(VAL_CHARS_re, sent)

    sent = sent.strip()

    sent = tag_CM_0.sub(tag_CM_0_re, sent)
    sent = tag_CM_1.sub(tag_CM_1_re, sent)
    sent = tag_CM_2.sub(tag_CM_2_re, sent)

    sent = tag_1.sub(tag_1_re, sent)
    sent = tag_2.sub(tag_2_re, sent)
    sent = tag_3.sub(tag_3_re, sent)
    sent = tag_4.sub(tag_4_re, sent)
    sent = tag_5.sub(tag_5_re, sent)
    sent = tag_6.sub(tag_6_re, sent)
    sent = tag_7.sub(tag_7_re, sent)
    sent = tag_8.sub(tag_8_re, sent)
    sent = tag_9.sub(tag_9_re, sent)
    sent = tag_10.sub(tag_10_re, sent)
    sent = tag_11.sub(tag_11_re, sent)
    sent = tag_12.sub(tag_12_re, sent)
    sent = tag_13.sub(tag_13_re, sent)
    sent = tag_14.sub(tag_14_re, sent)
    sent = tag_15.sub(tag_15_re, sent)
    sent = tag_16.sub(tag_16_re, sent)
    sent = tag_17.sub(tag_17_re, sent)
    sent = tag_18.sub(tag_18_re, sent)
    sent = tag_19.sub(tag_19_re, sent)
    sent = tag_20.sub(tag_20_re, sent)
    sent = tag_21.sub(tag_21_re, sent)
    sent = tag_22.sub(tag_22_re, sent)
    sent = tag_23.sub(tag_23_re, sent)
    sent = tag_24.sub(tag_24_re, sent)
    sent = tag_25.sub(tag_25_re, sent)
    sent = tag_26.sub(tag_26_re, sent)
    sent = tag_27.sub(tag_27_re, sent)
    sent = tag_28.sub(tag_28_re, sent)
    sent = tag_29.sub(tag_29_re, sent)
    sent = tag_30.sub(tag_30_re, sent)
    sent = tag_31.sub(tag_31_re, sent)
    sent = tag_32.sub(tag_32_re, sent)
    sent = tag_33.sub(tag_33_re, sent)
    sent = tag_34.sub(tag_34_re, sent)
    sent = tag_35.sub(tag_35_re, sent)
    sent = tag_36.sub(tag_36_re, sent)
    sent = tag_37.sub(tag_37_re, sent)
    sent = tag_38.sub(tag_38_re, sent)
    sent = tag_39.sub(tag_39_re, sent)
    sent = tag_40.sub(tag_40_re, sent)
    sent = tag_41.sub(tag_41_re, sent)
    sent = tag_42.sub(tag_42_re, sent)
    sent = tag_43.sub(tag_43_re, sent)
    sent = tag_44.sub(tag_44_re, sent)
    sent = tag_45.sub(tag_45_re, sent)
    sent = tag_46.sub(tag_46_re, sent)
    sent = tag_47.sub(tag_47_re, sent)
    sent = tag_48.sub(tag_48_re, sent)
    sent = tag_49.sub(tag_49_re, sent)
    sent = tag_50.sub(tag_50_re, sent)

    sent = sent.strip()

    sent = pat_1.sub(pat_1_re, sent)
    sent = pat_2.sub(pat_2_re, sent)
    sent = pat_3.sub(pat_3_re, sent)
    sent = pat_4.sub(pat_4_re, sent)

    sent = sent.strip()

    sent = rep_1.sub(rep_1_re, sent)
    sent = rep_2.sub(rep_2_re, sent)
    sent = rep_3.sub(rep_3_re, sent)
    sent = rep_4.sub(rep_4_re, sent)

    sent = sent.strip()

    if(not CORR_SENT.match(sent) and sent != ""):
        print(sent)
        return ""

    sent += '\n'

    return sent


def CleanBNCCorpus(inFileDir, outFileDir):
    # Create outFileDir
    if not os.path.exists(os.path.dirname(outFileDir)):
        try:
            os.makedirs(os.path.dirname(outFileDir))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(inFileDir, "r", encoding="utf_8") as inFile:
        with open(outFileDir, "w+", encoding="utf_8") as outFile:

            data = inFile.readlines()
            outFile.write("<bncDoc xml:id=\""+ outFileDir +"\">\n")

            for i in range(2, len(data)): outFile.write(CleanSent(data[i]))

            outFile.write("</bncDoc>\n")

def main():
    outFileDir = ROOT_DIR + OUT_SUF

    # Create New File Directory
    if not os.path.exists(os.path.dirname(outFileDir)):
        try:
            os.makedirs(os.path.dirname(outFileDir))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    for root, _, files in os.walk(ROOT_DIR):
        if files == []:
            continue

        print("Extracting Corpora in:", root)
        for corpus in files:
            inFileDir = os.path.join(root, corpus)
            outFileDir = inFileDir.replace(ROOT_DIR, ROOT_DIR + OUT_SUF)
            print(inFileDir)
            CleanBNCCorpus(inFileDir, outFileDir)

if __name__ == "__main__":
    main()
