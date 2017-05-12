#########################################
# George Saman                          #
# Character Recognition using ANN       #
# with backpropagation algorithm        #
# Email: georgesaman@csu.fullerton.edu  #
#########################################

from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt

# files i wrote
import CropAndNormalize as cAn
import LineExtraction as lN

########################################Letters To Train
letters = ['a','b','c','d','e','f','g','h','i',              # Letters to learn 
           'j','k','l','m','n','o','p','q','r',
           's','t','u','v','w','x','y','z']

##########################################################

        
#----------------------------------------------------------------------------------------------------
#------------------------------Weights initialization------------------------------------------------
#----------------------------------------------------------------------------------------------------
  
def initializeWeights(width,height,numberOfHiddenNeurons):
    Wi_h = np.random.random(size=(numberOfHiddenNeurons,height,width))-0.5  # input to hidden layer weights
    Wh_o = np.random.random(size=(26,numberOfHiddenNeurons))-0.5            # hidden to output weights
    Bh   = np.random.random(numberOfHiddenNeurons) - 0.5                    # hidden layer biases
    Bo   = np.random.random(26) - 0.5                                       # output layer biases    
    
    return Wi_h, Wh_o, Bh, Bo


#----------------------------------------------------------------------------------------------------
#------------------------------Threshold Function----------------------------------------------------
#----------------------------------------------------------------------------------------------------
def logistic(summation):                                                    # sigmoid function
       out = 1 / (1 + np.exp(-summation))
       return out

#----------------------------------------------------------------------------------------------------
#------------------------------FEED FORWARD THROUGH NETWORK------------------------------------------
#----------------------------------------------------------------------------------------------------
       
def feedForward(normalized, Wi_h, Wh_o, Bh, Bo):                            # feed forward through net
    
    n_h                                  = 0
    [numberOfHiddenNeurons,height,width] = Wi_h.shape
    #-----------------------------feed Forward, From input layer to hidden
    outputOfHiddenNeurons                = []
    netInputForHiddenNeurons             = []
    
    for hiddenNeuron in range (0,numberOfHiddenNeurons):                 # forward pass from input to hidden
        
        for i in range (0,height):                                       # calculating net activation input
            for j in range (0,width):                           
                WxP = Wi_h[hiddenNeuron,i,j] * normalized[i,j]           # Weight X Input 
                n_h   = n_h + WxP                                        # The overall sum of W's X P's
               
        n_h                   = n_h + Bh[hiddenNeuron]                   # total input = WP+Bias
        outputOfHiddenNeurons.append(logistic(n_h))                      # calculate and save hidden neurons output
        netInputForHiddenNeurons.append(n_h)                             # save total net input
        n_h                   = 0                                        # reset n

    #---------------------------feed forward, from hidden to output
    outHiddenXweightsH_O      = outputOfHiddenNeurons * Wh_o             # out of hidden layer multiplied by weights from hidden to output layer
                                                                         # this is a 26 X 10 matrix, each row contains the weights connecting hidden neurons to specific out neuron.
    netInputForOutNeurons     = np.sum(outHiddenXweightsH_O, axis= 1)    # sum of all (Wh_o weights X hidden neuron outputs), each row is the total input for each output neuron
    outputOfOutNeurons        = []
    
    for outputNeuron in range(0,26):
                                                                         # Find and Calculate output of out neurons
        totalInputForNeuron   = netInputForOutNeurons[outputNeuron] + Bo[outputNeuron]   # the input to the kth output neuron
        outputOfOutNeurons.append(logistic(totalInputForNeuron))         # get the output and save it

    return outputOfOutNeurons, outputOfHiddenNeurons                     # return outputs

#----------------------------------------------------------------------------------------------------
#------------------------------Calculate Error At Output Neurons-------------------------------------
#----------------------------------------------------------------------------------------------------
   

def calculateErrorAtOutput(outputOfOutNeurons, targetOutput):
    outputError =[]
    for outputNeuron in range(0,26):
                                                                         # Calculating the error at the output
        outputNeuronError     = outputOfOutNeurons[outputNeuron] - targetOutput[outputNeuron] # error = out - target
        outputError.append(outputNeuronError)                            # save error for all outputs
    return outputError                                                   # return error


#----------------------------------------------------------------------------------------------------
#------------------------------BACK PROPAGATE AND ADJUST WEIGHTS ------------------------------------
#----------------------------------------------------------------------------------------------------
   
def backPropagate(Wi_h, Wh_o, Bh, Bo, normalized, outputError, outputOfOutNeurons, outputOfHiddenNeurons, learningRate, momentum):

    oldWh_o    = np.array(Wh_o[:,:])                                     # save old weights for b.prop from hidden to input  
    oldWi_h    = np.array(Wi_h[:,:]) 
   
    [numberOfHiddenNeurons,height,width] = Wi_h.shape
    #---------------------------------------Back Propagating from output to hidden and adjusting weights
    for outputNeuron in range(0,26):
        for hiddenNeuron in range(0,numberOfHiddenNeurons):
            # calculating the adjustment which is learning rate * error at current output neuron * sigmoid derivative *  output of current hidden neuron)
            adjustment                       = (learningRate * outputError[outputNeuron] * outputOfOutNeurons[outputNeuron] * (1 - outputOfOutNeurons[outputNeuron]) * outputOfHiddenNeurons[hiddenNeuron])
            Wh_o[outputNeuron, hiddenNeuron] = (momentum * Wh_o[outputNeuron, hiddenNeuron]) - adjustment # adjusting weights per this formula, Wnew = momentum* Wold - adjustment

    #---------------------------------------Back Propagating from hidden to input and adjusting weights
    for hiddenNeuron in range(0,numberOfHiddenNeurons):
        deltaTotalError_hiddenNeuron = 0
        for outputNeuron in range(0,26):
            # Calculate delta error at each output neuron with respect to current hidden neuron
            deltaErrorOutputNeuron_hiddenNeuron = outputError[outputNeuron] *  outputOfOutNeurons[outputNeuron] * (1-outputOfOutNeurons[outputNeuron]) * oldWh_o[outputNeuron,hiddenNeuron]
            deltaTotalError_hiddenNeuron = deltaTotalError_hiddenNeuron + deltaErrorOutputNeuron_hiddenNeuron # delta total error with respect to current hidden neuron

        # loop over all input weights connecting to current hidden neuron
        for i in range (0,height):
            for j in range (0,width):
                # delta Total Error with respect to weight to be adjusted. this weight is connecting input to current hidden layer
                deltaTotalError_inputTohiddenNeuronWeight = deltaTotalError_hiddenNeuron * outputOfHiddenNeurons[hiddenNeuron] *(1 - outputOfHiddenNeurons[hiddenNeuron]) * normalized[i,j]
                Wi_h[hiddenNeuron,i,j] = (momentum * Wi_h[hiddenNeuron,i,j]) - (learningRate * deltaTotalError_inputTohiddenNeuronWeight)

    return Wi_h, Wh_o

#----------------------------------------------------------------------------------------------------
#------------------------------TRAIN NETWORK --------------------------------------------------------
#----------------------------------------------------------------------------------------------------
   
def trainNet(Wi_h, Wh_o, Bh, Bo,height,width,numberOfTrainingSamples,learningRate,momentum,targetError):
    iteration  = 0
    totalError = 1
    errorList  = []     # to save all total error generated
    y_axis     = []     # for plotting the error minimization at the end
        
    while totalError > targetError:                                            # loop until criteria is met

        for letterToTrain in range(0,26):                                      # loop for all letters to be trained 
            targetOutput                = np.zeros(26)                         # target output is all zeros
            targetOutput[letterToTrain] = 1                                    # except the one to be trained
            
            for n in range (0,numberOfTrainingSamples):                        # number of training samples 

                #---------------Cropping and Normalizing the image to have a uniform input to the ANN
                trainingSample      = 'samples/%s%d.png' %(letters[letterToTrain],n) # training sample image file name

                character_in        = im.open(trainingSample)                  # load Image
                blackAndWhite       = cAn.convertToBW(character_in)            # Convert to BW, 1->black
                toggledBW           = cAn.toggleOnesAndZeros(blackAndWhite)
                croppedBW           = cAn.crop(toggledBW)                      # Crop Image to get character only
                normalized          = cAn.normalize(croppedBW,width,height)    # Normalize (resize)
               #----------------------- end of pre processing phase
                
                outputOfOutNeurons, outputOfHiddenNeurons = feedForward(normalized, Wi_h, Wh_o, Bh, Bo) # feed forward
                outputError         = calculateErrorAtOutput(outputOfOutNeurons, targetOutput)          # calculate error at output neurons
                Wi_h, Wh_o          = backPropagate(Wi_h, Wh_o, Bh, Bo, normalized, outputError, outputOfOutNeurons, outputOfHiddenNeurons, learningRate,momentum) # backpropage and adjust weights

        #-----------------------Calculate the mean squared error
        totalError = 0
        for x in range(0,26):
            squared     = 0.5 * outputError[x]**2
            totalError  = totalError + squared
            
        print('Total Error = %f' %totalError)
        iteration = iteration + 1
        errorList.append(totalError)
        y_axis.append(iteration)
        
    #------------------Plot Total Error vs Iteration
    print('Total Number of iterations %d' %iteration)
    plt.plot(y_axis, errorList)
    plt.ylabel('Total Error')
    plt.xlabel('Number Of Iterations')
    plt.show()
    
    return (Wi_h, Wh_o, Bh, Bo)


#----------------------------------------------------------------------------------------------------
#------------------------------Recognize Character---------------------------------------------------
#----------------------------------------------------------------------------------------------------               
    
def recognizeCharacter(inputNormalized,Wi_h,Wh_o,Bh,Bo):                # Returns the character recognized 
   
    outputOfOutNeurons, outputOfHiddenNeurons = feedForward(inputNormalized, Wi_h, Wh_o, Bh, Bo) # feed forward
    maxOut                                    = np.argmax(outputOfOutNeurons)                    # character recognized is neuron with highest output
    return letters[maxOut]
