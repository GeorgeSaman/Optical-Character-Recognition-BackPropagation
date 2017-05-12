#########################################
# George Saman                          #
# CROP AND NORMALIZATION                #
# Email: georgesaman@csu.fullerton.edu  #
#########################################

import numpy as np
from PIL import Image as im

def convertToBW(imageIn):
    blackAndWhite                          = imageIn.convert('1')           # 0 -> black , 1 -> white
    blackAndWhite                          = np.array(blackAndWhite)*1      # convert to array
    return blackAndWhite                                                    # 0 -> black , 1 -> white


def toggleOnesAndZeros(blackAndWhite):                                      # 1 -> black , 0 -> white
    return (blackAndWhite ^1)
    

def modifyInputPixelsValues(blackAndWhite):                                 # Assign +3 -> black 
    in_blackAndWhite = blackAndWhite
    [numberRowPixels , numberColumnPixels] = blackAndWhite.shape            # find array dimensions
   
    for i in range (0,numberRowPixels):                                                                         
        for j in range (0,numberColumnPixels):                                  
            if in_blackAndWhite[i,j] >0 :                                   
                in_blackAndWhite[i,j] = 3

    toggled = in_blackAndWhite
    return toggled



                
def crop(blackAndWhiteToggled):
    
    [numberOfRowPixels , numberOfColumnPixels] = blackAndWhiteToggled.shape              # find array dimensions
    
    #-----------------------------------------Finding the left and right side
    verticalSumOfBlackPixels                   = np.sum(blackAndWhiteToggled,axis=0)     # gives a list of number of black pixels in each column
    leftDetected = False
    for i in range(0,numberOfColumnPixels):
        if verticalSumOfBlackPixels[i] > 0 and leftDetected == False:             # there is a black pixel in this column
            leftDetected = True
            left = i                                                              # left
        elif verticalSumOfBlackPixels[i] > 0 and leftDetected == True:
            right = i                                                             # right

		
    #-----------------------------------------Finding the top and bottom side
    horizontalSumOfBlackPixels                   = np.sum(blackAndWhiteToggled,axis=1)   # gives a list of number of black pixels in each row
    topDetected = False
    for i in range(0,numberOfRowPixels):
        if horizontalSumOfBlackPixels[i] > 0 and topDetected == False:            # there is a black pixel in this column
            topDetected = True
            top = i                                                               # top
        elif horizontalSumOfBlackPixels[i] > 0 and topDetected == True:
            bottom = i                                                            # bottom

        
    v_CroppedBlackAndWhite_array  = blackAndWhiteToggled[:,(range(left,right+1))]
    finalCroppedBlackAndWhite     = v_CroppedBlackAndWhite_array[(range(top,bottom+1)),:]
    # Transform array back to image
    return finalCroppedBlackAndWhite


    
def normalize(character_in,width,height):
    
    # Why Hamming?
    # Produces more sharp image than BILINEAR and doesnot have dislocations on local level like BOX.
    character_in   = im.fromarray(character_in)
    normalized     = character_in.resize((width,height),im.HAMMING)         # normalize image to desired dimensions
    NormalizedArray    = np.array(normalized) 
    return NormalizedArray                                                   # Output



##
##
##inputImageFileName      = 'letters/l.png'
##inputImage              = im.open(inputImageFileName)                       # load input image
##BW                      = convertToBW(inputImage)                       # convert to black and white
##toggled                 = toggleOnesAndZeros(BW)                        # make 1-> black, 0-> white
##inputCroppedBW          = crop(toggled)
##inputCroppedBWtoggled   = toggleOnesAndZeros(inputCroppedBW)
##colored                 = inputCroppedBWtoggled*255
##img                     = im.fromarray(colored)
##inputImage.show()
##img.show()
###modifiedBW              = modifyInputPixelsValues(inputCroppedBW)


