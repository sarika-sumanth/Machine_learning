import cv2
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
image=cv2.imread("temp/no_noise.jpg")
string=pytesseract.image_to_string(image)
print(string)



# RESULT
# “GABRIEL Meamall

# On Easter morning in the year 194%, I took my six-year-old
# son by the hand and began walking fron my home town toward the
# valleys and forests of the Carpathi2n mountains. For nearly
# eight months we lived in barns, attics and makeshift eabins. With
# the generous nelp of an unusually courageous man, we managed to
# survive Europe's greatest fit of madness. Those who walked in
# the opposite direction on that Easter day were lese fortunate.
# They were taken in trainloads to places whose once obscure names
# are now, and forever will be, synonymous with terror, evil and
# death. What follows is our story of survival told to the best
# of my ability, in plain, simple language.

# In March of 1944 the SS troops took over the internal affairs
# of Hungary and proceeded to organize the deportation of the dows.
# To the Nazie thie was a routine assignment; within hours all local
# officials were informed of operational plans. The high command
# issued a directive designed to placate Jewish fears and induce
# cooperation. It was announced that the Jews would be shipped to
# Poland as an emergency labor force and that they were only being
# drafted for temporary work. There were many who believed this
# version. Others, less credulous, resigred themselves and hoped for
# the vest. Still others began to make plans for escape. By Aprii
# 13 the Hungarian Jews were being rounded up from all over the .
# country in what was once a huge brick factory. The rest is well
# known. .

# I was working in Ungvar and usually came home on weekends.

# At that time it was no longer possible for a Jew to travel freely.

