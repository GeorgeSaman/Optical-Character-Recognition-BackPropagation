#########################################
# George Saman                          #
# Line Extraction                       #
# Email: georgesaman@csu.fullerton.edu  #
#########################################

from PIL import Image as im
import numpy as np
import CropAndNormalize as cAn


''' returns the topOfline,bottomOfLine list of all lines 
Examples
 000000000000000   -> whiteSpace
 101010000000011   -> topOfLine
 101010101000110   -> line content
 101000101111010   -> bottomOfline
 000000000000000   -> whiteSpace
'''

#############################################################
################# CROPPING PARAGRAPHS FROM DOCUMENT #########
    

def cropParagraphs(numberOfLines,topOfLines,bottomOfLines):
    locationOfNewLines    = []
    for i in range(1,numberOfLines):
        whiteSpaceDistance = topOfLines[i] - bottomOfLines[i-1]
        if whiteSpaceDistance > 60:                             # White space pixels must be more than 60 pixel to be considered a new paragraph
            locationOfNewLines.append(i-1)                      # -1 because python indexing starts with a zero
    return locationOfNewLines

#############################################################
################# CROPPING LINES FROM PARAGRAPHS ############

def cropLines(blackAndWhite):                                             # of black pixels in a row

    [numberRowPixels , numberColumnPixels] = blackAndWhite.shape    
    h_firstBlackPixelDetected = False
    firstBlackPixelRow  = 0
    lastBlackPixelRow   = 0
    topOfLines          = []
    bottomOfLines       = []

    for i in range (0,numberRowPixels):
        sumOfAllPixelsInRow_i = sum(blackAndWhite[i,:])
        if sumOfAllPixelsInRow_i >= 1 and h_firstBlackPixelDetected == False: # Detects First horizontal Row
            h_firstBlackPixelDetected = True
            firstBlackPixelRow = i
            lastBlackPixelRow  = i
            
        elif sumOfAllPixelsInRow_i >= 1 and h_firstBlackPixelDetected == True:# Detects Last Horizontal Row
            lastBlackPixelRow  = i

         
        elif sumOfAllPixelsInRow_i < 1 and h_firstBlackPixelDetected == True: # Detects a White row
            h_firstBlackPixelDetected = False
            topOfLines.append(firstBlackPixelRow)                            # Save firstBlackPixels in a list
            bottomOfLines.append(lastBlackPixelRow)                          # Save LastBlackPixels in a list

    # Creating a list that contains all cropped Lines 
    numberOfLines       = len(topOfLines)  
    croppedLinesList                          = []
    for i in range(0,numberOfLines):                                         # Make a list containing croppedLines
        croppedLine         = blackAndWhite[(range(topOfLines[i],bottomOfLines[i])),:]
        croppedLinesList.append(croppedLine)                    

    return croppedLinesList,numberOfLines,topOfLines,bottomOfLines


#############################################################
################# CROPPING CHARACTERS FROM LINE #############

def cropCharacters(croppedLine,numberOfLines):
    
    [numberOfLineRowPixels , numberOfLineColumnPixels] = croppedLine.shape    
    leftDetected        = False
    firstBlackPixelColumn  = 0
    lastBlackPixelColumn   = 0
    leftOfCharacters    = []
    rightOfCharacters   = []

    for i in range (0,numberOfLineColumnPixels):
        sumOfAllPixelsInColoumn_i = sum(croppedLine[:,i])
        if sumOfAllPixelsInColoumn_i >= 1 and leftDetected == False:          # left found
            leftDetected = True
            firstBlackPixelColumn = i
            lastBlackPixelColumn  = i
            
        elif sumOfAllPixelsInColoumn_i >= 1 and leftDetected == True:         # Detects Last Horizontal Row
            lastBlackPixelColumn  = i

         
        elif sumOfAllPixelsInColoumn_i < 1 and leftDetected == True: # Detects a White row
            leftDetected = False
            leftOfCharacters.append(firstBlackPixelColumn)                          # Save left sides of characters in a list
            rightOfCharacters.append(lastBlackPixelColumn)                          # Save right sides of characters in a list

          # Creating a list that contains all cropped Lines 
    numberOfCharacters                             = len(leftOfCharacters)  
    croppedCharactersList                          = []
    for i in range(0,numberOfCharacters):                                         # Make a list containing croppedLines
        croppedCharacter        = croppedLine[:,(range(leftOfCharacters[i],rightOfCharacters[i]))]
        croppedCharactersList.append(croppedCharacter)
        
    return croppedCharactersList,numberOfCharacters,leftOfCharacters,rightOfCharacters

#############################################################
################# CROPPING WORDS FROM LINE ##################
    
def cropWords(recognizedCharactersList,leftOfCharacters,rightOfCharacters):
    numberOfCharacters      = len (leftOfCharacters)
    locationOfSpaces        = [0]
    words                   = []
    for i in range(1,numberOfCharacters):
        whiteSpaceBetweenCharacters  = leftOfCharacters[i] - rightOfCharacters[i-1]
        if whiteSpaceBetweenCharacters > 20 :
            locationOfSpaces.append(i)
        if i == numberOfCharacters-1:
            locationOfSpaces.append(numberOfCharacters)
    for i in range(0,len(locationOfSpaces)-1):
        firstCharacterInWord = locationOfSpaces[i]
        lastCharacterInWord  = locationOfSpaces[i+1]
        word                 = recognizedCharactersList[firstCharacterInWord:lastCharacterInWord]
        word                 = ''.join(word)
        words.append(word)
    numberOfWords           = len(words)
    return words,numberOfWords
        

    


