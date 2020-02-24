import random
import numpy as np
import time
from collections import defaultdict
import csv
DIFF = 4000


class mine:
    def __init__(self,k,power):
        self.k=k
        self.power=power
        self.bits=64
        self.vals=[2**self.bits]
        t1=2**self.bits/DIFF
        self.target= float(t1*(self.k+1)/2)

    def should_mine(self):
        return random.random() < self.power

    def val_mean(self):
        return (sum(self.vals)/len(self.vals))

    def mine_success(self):
        """ returns success boolean and values """  
        val = random.getrandbits(self.bits)
        if val<= max(self.vals):
            self.vals=sorted(self.vals+[val])[0:self.k]
        if np.mean([float(x) for x in self.vals])<self.target:
            return(True)
        else:
            return(False)

def sim_noec(q,z,k):
    honest = mine(k,max(0,1-q))
    attacker= mine(k,q)
    
    first_honest= mine(k,max(0,1-q))
    first_attacker = mine(k,q)
    
    honest_length = 0
    attacker_length = 0
    att_hashes = 0
    honest_hashes=0

    while(True):
        dice = np.random.uniform(0,1)
        if (dice<q):
          att_hashes += 1
          if (attacker_length==0):
            #attacker still hasn't mined their first block
            first_attacker.mine_success()
            #check block success 
            # use only honest proof values that are larger than attacker's 1OS
            min_attacker= min(first_attacker.vals)
            hvals = [x for x in first_honest.vals if x> min_attacker]
            vals = sorted(first_attacker.vals+hvals)[0:k]
            if np.mean(vals)<attacker.target:
              #this increment causes attacker to start mining on their own fork
              attacker_length=1
          else:
            #attack is on second block, normal sim
            if attacker.mine_success():
              #we mined a block, so reset and add to length
              attacker.vals=[2**attacker.bits]
              attacker_length += 1
        else:
          honest_hashes+=1
          if (honest_length==0):
            #honest still hasnt mined thei first sim
            first_honest.mine_success()
            #check block success
            #ignore the lowest vals if they from the attacker and check for block
            min_honest= min(first_honest.vals)
            #only use attacker values that are larger than honests' 1OS
            avals = [x for x in first_attacker.vals if x> min_honest]
            vals = sorted(avals+first_honest.vals)[0:k]
            if np.mean(vals)<first_honest.target:
              honest_length=1

          else:
            if honest.mine_success():
              #we mined a block, so reset and add to length
              honest.vals=[2**honest.bits]
              honest_length += 1          
          
        if honest_length >= z+1500+3*z:
            return ((False, att_hashes,honest_hashes))
        if (attacker_length >= z+1) and (attacker_length>honest_length):
            return ((True, att_hashes,honest_hashes))



t=int(time.time())
r=np.random.randint(0,1000000)

agg=defaultdict(set)

outfile=open('agg.rand=%d%d.csv' % (t,r), 'w')
for run in range(0,5):
    q= random.choice([.1,.2,.3,.4,.45])
    z= random.choice([0,1,2,3,4,5,6,7,8,9,10])
    k= random.choice([1,2,10,20,30,40,50,60,100]) #np.arange(1,40,4):
    wins,att_hashes,hon_hashes=sim_noec(q,z,k)
    outfile.write("%s, %s, %s, %s,%s, %s, %s\n" %(DIFF,q,z,k,wins,att_hashes,hon_hashes))
    outfile.flush()
outfile.close()
