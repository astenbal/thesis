# Class that contains all result values from different datasets
class ResultSet():
    trueAnswer = False
    sampleAnswer = False
    unfilteredAnswer = False
    sampleUnfilteredAnswer = False

    # Fill member variables in order
    def FillNext(self, value):
        if(not self.trueAnswer):
            self.trueAnswer = value
        elif(not self.sampleAnswer):
            self.sampleAnswer = value
        elif(not self.unfilteredAnswer):
            self.unfilteredAnswer = value
        elif(not self.sampleUnfilteredAnswer):
            self.sampleUnfilteredAnswer = value
        else:
            print('All values have been submitted')