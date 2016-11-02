import numpy as np
from scipy.ndimage import imread

class BatchBuffer:
    def __init__(self, offset, size):
        self.offset = offset
        self.size = size
        self.index = 0
    
    def read(self, num=None):
        raise NotImplementedError('Unimplemented')
        
    def seek(self, num, mode=0):
        if mode == 0:
            self.index = num
        elif mode == 1:
            self.index += num
        elif mode == 2:
            self.index = self.size+num
        else:
            raise NotImplementedError('Unimplemented')
        
    def tell(self):
        return self.index

class ArrayBuffer(BatchBuffer):
    def __init__(self, array, offset, size):
        if size < 0:
            size = len(array) - offset
        super().__init__(offset, size)
        self.array = array
    
    def read(self, num=None):
        if self.index >= self.size:
            return None
        
        if num is None:
            num = self.size - self.index
            
        ni = self.index+num
        if ni >= self.size:
            ni = self.size
        ret = self.array[self.offset+self.index:self.offset+ni]
        self.index += num
        return ret
    
class TarBuffer(BatchBuffer):
    def __init__(self, tar, offset, size):
        self.tar = tar
        self.members = tar.getmembers()[offset:-1]
        if size < 0:
            size = len(self.members)
        else:
            self.members = self.members[:size]
        super().__init__(offset, size)
        
        self.i = 0
    
    def read(self, num=None):
        if self.index >= self.size:
            return None
        
        if num is None:
            num = self.size - self.index
            
        ni = self.index+num
        if ni >= self.size:
            ni = self.size
            
        #start_time = time.time()
            
        images = [1 - (imread(self.tar.extractfile(self.members[self.index+i]))/255) for i in range(num)]
        ret = np.array(images)
        self.index += num
        
        #self.i += 1
        #print ("%3dth, %ss" % (self.i, time.time()-start_time))
        
        return ret