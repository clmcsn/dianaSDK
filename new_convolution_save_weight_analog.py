def convolution_save_weights_analog(self,index_layer):
        Fx = self.nnConfig.Fx[index_layer]
        Fy = self.nnConfig.Fy[index_layer]
        C = self.nnConfig.C[index_layer]
        K = self.nnConfig.K[index_layer]

        self.mapWeight(index_layer)
  
        for idx in range(len(self.aniaMem)):
            for row in range(self.usedRow[idx]):
                #TODO try the way around
                #write crossbar is "mirrored"! Weights order must be flipped!
                for col in range(self.usedCol[idx]-1,-1,-1):
                    if self.aniaMem[idx][row][col] == None:
                        self.file_w_cnn_ania.addPadded()
                    else:
                        block = row//64
                        if ((col%2) and (block in [6, 7, 8, 9, 10, 11])): #if odd column
                            data = -self.aniaMem[idx][row][col][0]
                        elif ((col%2==0) and (block>8)): #if even column
                            data = -self.aniaMem[idx][row][col][0]
                        else:
                            data = self.aniaMem[idx][row][col][0]
                        self.file_w_cnn_ania.addWeight(ternary_to_hex(data),index_layer, self.aniaMem[idx][row][col][1],self.aniaMem[idx][row][col][2],self.aniaMem[idx][row][col][3],self.aniaMem[idx][row][col][4])

