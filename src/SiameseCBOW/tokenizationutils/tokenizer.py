# -*- coding: utf-8 -*-
import re

# With period
sNonTokenChars = u"[‘’“”…”’“–«»\.\,‘\]\[;:\-\"'\?!¡¢∞§¶•ª≠∑´®†¨^πƒ©˙∆˚¬≈√∫~⁄™‹›ﬁﬂ‡°·±—‚„‰∏”`◊ˆ~¯˘¿÷\*\(\)<>=\+#^\\\/_]+"
# Without period
sNonTokenChars_noPeriod = u"[‘’“”…”’“–«»\,‘\]\[;:\-\"'\?!¡¢∞§¶•ª≠∑´®†¨^πƒ©˙∆˚¬≈√∫~⁄™‹›ﬁﬂ‡°·±—‚„‰∏”`◊ˆ~¯˘¿÷\*\(\)<>=\+#^\\\/_]+"

reNonTokenChars_start = re.compile(u"(\A|\s)%s" % sNonTokenChars, re.U)
reNonTokenChars_end = re.compile(u"%s(\s|\Z)" % sNonTokenChars, re.U)
reNonTokenChars_end_noPeriod = \
    re.compile(u"%s(\s|\Z)" % sNonTokenChars_noPeriod, re.U)

reWhitespace = re.compile("\s+", re.U)

def removeNonTokenChars(sString):
  sString = re.sub(reNonTokenChars_start, '\g<1>', sString)
  return re.sub(reNonTokenChars_end, '\g<1>', sString)

def tokenizeSentence(sString, bLowerCase=True):
  sString = sString.replace("/", " / ")

  aTokens = None
  if bLowerCase:
    aTokens = reWhitespace.split(removeNonTokenChars(sString.lower()))
  else:
    aTokens = reWhitespace.split(removeNonTokenChars(sString))

  # split() gives empty first/last elements if there were separators at the
  # start/end of the string. We correct for that.
  iStart = 1 if aTokens[0] == '' else 0
  if aTokens[-1] == '':
    return aTokens[iStart:-1]
  else:
    return aTokens[iStart:]
