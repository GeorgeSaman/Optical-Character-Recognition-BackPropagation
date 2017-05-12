import LineExtraction as lN
import CharacterRecognition as cR
import CropAndNormalize as cAn
from PIL import Image as im

#####PARAMETERS
learningRate = 0.5; momentum = 1; targetError = 0.0035 ;numberOfHiddenNeurons = 80
width=18; height = 16 ;
numberOfTrainingSamples = 4;
documentLocation        = 'paragraphs/ocr1.png'

######################################################################################
############################## Prepare Image #########################################
print('Loading Image....')
imageIn                                                 = im.open('%s' %documentLocation)          # open image
imageInBW                                               = cAn.convertToBW(imageIn)                 # 0-> black , 1-> white
imageInBW                                               = cAn.toggleOnesAndZeros(imageInBW)        # 1-> black , 0 -> white
imageIn.show()
print('Image Loaded')
######################################################################################
############################## Initialize And Train Network ##########################

print('Training In Progress.....')
Wi_h, Wh_o, Bh , Bo = cR.initializeWeights(width,height,numberOfHiddenNeurons)                     # initialize weights
Wi_h, Wh_o, Bh , Bo = cR.trainNet(Wi_h, Wh_o, Bh , Bo,height,width,numberOfTrainingSamples,learningRate,momentum,targetError) # train net
print('Neural Net Trained')


######################################################################################
##################### Recognize Paragraphs,Lines,Words and Characters#################

[croppedLinesList,numberOfLines,topOfLines,bottomOfLines]  = lN.cropLines(imageInBW)              # Extract Lines
locationOfNewLines                                         = lN.cropParagraphs(numberOfLines,topOfLines,bottomOfLines) # get location of new lines
numberOfNewLines                                           = len(locationOfNewLines)
linesContents                                              = []

# Loop for all Lines
for line in range(0,numberOfLines):
    [croppedCharactersList,numberOfCharacters,leftOfCharacters,rightOfCharacters]= lN.cropCharacters(croppedLinesList[line],numberOfLines)# Characters from Line
    recognizedCharacterlist        = []
    # Loop for all characters in lines
    for character in range(0,numberOfCharacters):
        inputCroppedBW          = cAn.crop(croppedCharactersList[character])                    # crop image to get the character only
        inputNormalized         = cAn.normalize(inputCroppedBW,width,height)                    # normalize to neural net size
               
        output                  = cR.recognizeCharacter(inputNormalized,Wi_h, Wh_o, Bh, Bo)     # normalized image is sent to recognition
        recognizedCharacterlist.append(output)                                                  # save characters found in line

        print('character number %d is %s' %(character,output))
       
    #crop Words from Line
    [words,numberOfWords] = lN.cropWords(recognizedCharacterlist,leftOfCharacters,rightOfCharacters) # form words from characters found in line
    linesContents.append(words)                                                                 # save words
     
print(linesContents)

#######################################################################################
################################# STORE TO A TXT ######################################

fileName = open('OCR_OUTPUT.txt','w')
newLinesIndex = 0
for line in range(0,len(linesContents)):
    if line > 0:                                    # No NewLine if its the first line
        fileName.write('\n')
    for word in range(0,len(linesContents[line])):  # writing words
        fileName.write(linesContents[line][word])
        fileName.write(' ')
    
    if line == locationOfNewLines[newLinesIndex]:   # paragraphs spacing
        fileName.write('\n')
        if numberOfNewLines-1 > newLinesIndex:
            newLinesIndex += 1
fileName.close()






