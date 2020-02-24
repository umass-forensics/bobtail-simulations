import random
import numpy as np
import heapq
from scipy.stats import gamma

# For simplicity, we've placed all sims in this one file. 
# These methods are meant to run on a massively parallel system
# such as a slurm-based cluster, but can be run on a single machine
# Uncomment one of the four methods at the end of this file to run
# a simulation. 


global fd

PROOF=0
MINER=1
LOS = 2
TIME = 3
PUBLICIZED = 4
HONEST = "honest"
ATTACKER = "attacker"



class sim:
  def __init__(self,k=10):
    self.k=k
    self.bits= 32
    self.theta = 4000
    self.beta = 2**self.bits / self.theta
    self.target = self.beta*(self.k+1)/2
    assert self.k*self.target<2**self.bits, "k too high given beta"
    self.num_blocks_mined=0
    self.n = 1
    self.last_n_cnts = [self.theta]*self.n
    self.last_n_targets = [self.target]*self.n

  def check_for_block(self,vals,miner_idx):
    """ 
    This method ensures that 1) the block can be released only
    by the owner of the 1os (as if it's signed); 
    2) all proofs reference a support that is the 1os or larger.
    """
    FOS = min(vals,key=lambda x:x[PROOF])
    if FOS[MINER]!=miner_idx:
      return None
    filtered_vals = [x for x in vals if x[LOS]>= FOS[PROOF]]

    if len(filtered_vals)<self.k:
      #not enough values; no block
      return(None)
    k_values = heapq.nsmallest(self.k,filtered_vals,key=lambda x: x[PROOF])

    if np.mean([x[PROOF] for x in k_values]) > self.target:
      return(None)
    return(k_values)


  def pack_block_miner(self,vals,miner_idx):
    """ In this scenarios, if this miner is the FOS, it packs the block; otherwise returns none """
    FOS = min(vals,key=lambda x:x[PROOF])
    if FOS[MINER]!=miner_idx:
        return None
    packer_vals = [x for x in vals if x[MINER]==miner_idx]
    packer_vals = heapq.nsmallest(self.k,packer_vals,key=lambda x:x[PROOF])
    others_vals = [x for x in vals \
      if x[MINER]!=miner_idx and \
         x[PROOF]< packer_vals[-1][PROOF] and\
         x[LOS] >= packer_vals[0][PROOF]]
    others_vals =  sorted(others_vals,key=lambda x:x[PROOF])
    while (self.check_for_block(packer_vals,miner_idx) is None and len(others_vals)>0):
      packer_vals.append(others_vals.pop(0))
    return (self.check_for_block(packer_vals,miner_idx))

  def pack_block_miner_othersFIFO(self,vals,miner_idx):
    """ if this miner is the FOS, it packs the block; otherwise returns none.
    The proofs of other miners are packed in order they are broadcast """
    FOS = min(vals,key=lambda x:x[PROOF])
    if FOS[MINER]!=miner_idx:
        return None
    packer_vals = [x for x in vals if x[MINER]==miner_idx]
    packer_vals = heapq.nsmallest(self.k,packer_vals,key=lambda x:x[PROOF])
    others_vals = [x for x in vals \
      if x[MINER]!=miner_idx and \
         x[PROOF]> packer_vals[0][PROOF] and\
         x[LOS] >= FOS[PROOF]]
    others_vals =  sorted(others_vals,key=lambda x:x[PUBLICIZED])
    #
    block = self.check_for_block(packer_vals,miner_idx)
    while (block is None and len(others_vals)>0):
      new_val = others_vals.pop(0)
      packer_vals.append(new_val)
      block = self.check_for_block(packer_vals,FOS[MINER])
    return (block)



  def calculate_rewards(self,block,num_miners):
    """ return arrays representing rewards for proofs and bonuses """
    if block is None:
      return 0
    proofs = [0]*num_miners
    bonuses = [0]*num_miners
    FOS = min(block,key=lambda x:x[PROOF])
    for p in block:
      proofs[p[MINER]]+=1
      if p[LOS]==FOS[PROOF]:
        bonuses[p[MINER]]+=1
    return(proofs,bonuses)


  ######
  # The following methods represent the various scenarios considered 
  # in the paper. Although there is a lot of common code, it was easier
  # to code each as a different method

  def run_all_honest(self,attacker_power):
    """ 
    Everyone is honest and we just mine. 
    """
    global RANDOMNUM, fd
    public_vals = []
    total_hash_cnt = 0
    miners = [HONEST,HONEST]
    while (True):
      val= random.getrandbits(self.bits)
      total_hash_cnt+=1
      if val> self.k*self.target:
        continue
      mining_power = [attacker_power,(1-attacker_power)]
      miner_idx = np.random.choice(2,p=mining_power)

      los = min([x[PROOF] for x in public_vals]+[2**self.bits]) #least order stat
      val_record = [val,miner_idx,los,total_hash_cnt] #make a record of the new value
      heapq.heappush(public_vals,val_record)
      public_vals=heapq.nsmallest(self.k*8,public_vals)
      public_block= self.check_for_block(public_vals,miner_idx)
      
      if public_block is not None: #we found a block
        proofs, bonuses = self.calculate_rewards(public_block,len(miners))
        for idx,strategy in enumerate(miners):
          fd.write("%d, %f,%s, %s, %s, %s, %d\n" % (total_hash_cnt,\
          mining_power[idx], "no-withholding" , strategy, proofs[idx],bonuses[idx],self.k))
        return()

  def run_all_honest_count(self,thresh):
    """ 
    In this scenario, everyone is honest and we just mine. 
    We count traffic the amount of traffic generated, and 
    therefore output CSV is different than other methods
    """
    global RANDOMNUM, fd
    public_vals = []
    total_hash_cnt = 0 
    kt_count = 0 #amount of traffic below k*t bound
    smallestk_count = 0  # amount of traffic when proof is among smallest seen (and below k*t)
    both_count =0
    gamma_count = 0 
    max_gamma = 0
    miners = [HONEST]
    mining_power = [1]
    prop_delay = self.theta/600*10
    while (True):
      val= random.getrandbits(self.bits)
      total_hash_cnt+=1
      if val<= self.k*self.target:
        kt_count+=1

      mempool= [x for x in public_vals if ((total_hash_cnt-x[TIME])>=prop_delay) ]
      
      los = min([x[PROOF] for x in public_vals]+[2**self.bits])
      val_record = [val,0,los,total_hash_cnt]
      heapq.heappush(public_vals,val_record)
      public_vals=heapq.nsmallest(self.k,public_vals)
      
      if len(mempool)>=self.k:
        largest = heapq.nsmallest(self.k,mempool)[-1]
        if val < largest[PROOF]:
          smallestk_count +=1
          if val <= self.k*self.target:
            both_count+=1
      else:
        smallestk_count +=1
        if val <= self.k*self.target:
          both_count+=1
      gamma_val = gamma.cdf(val,a=self.k,scale=(2**self.bits)/self.theta)
      if gamma_val<=thresh:
        gamma_count+=1
      
      
      public_block= self.check_for_block(public_vals,0)
      if public_block is not None: #we found a block
        max_gamma = gamma.cdf(public_block[-1][PROOF],a=self.k,scale=(2**self.bits)/self.theta)
        fd.write("%d, %f, %d, %d, %d, %d, %d, %f\n" % (self.k, thresh, total_hash_cnt, kt_count, smallestk_count, both_count, gamma_count, max_gamma))
        return()



  def run_all_honest_winner_packs(self,attacker_power):
    """ In this scenario, a variation of run_all_honest, we 
    allow the winning miner to pack the block with their own proofs.
    The major difference is a call to self.pack_block_miner() when
    creating the block.
    """
    
    public_vals = []
    total_hash_cnt = 0
    miners = [HONEST,HONEST,HONEST]
    while (True):
      val= random.getrandbits(self.bits)
      total_hash_cnt+=1
      if val> self.k*self.target:
        continue
      mining_power = [attacker_power,(1-attacker_power)/2,(1-attacker_power)/2]
      miner_idx = np.random.choice(3,p=mining_power)

      los = min([x[PROOF] for x in public_vals]+[2**self.bits])
      val_record = [val,miner_idx,los,total_hash_cnt]
      heapq.heappush(public_vals,val_record)
      public_vals=heapq.nsmallest(self.k*8,public_vals)
      public_block= self.pack_block_miner(public_vals,miner_idx)
      if public_block is not None: #we found a block
        proofs, bonuses = self.calculate_rewards(public_block,len(miners))
        for idx,strategy in enumerate(miners):
          fd.write("%d, %f,%s, %s, %s, %s, %s\n" % (total_hash_cnt,\
                    mining_power[idx],
                    "winner_packs" , strategy,
                    proofs[idx],bonuses[idx],self.k))
          return()
  
  
  def run_withhold(self,attacker_power):
    """ In this scenario, the winning miner maliciously withholds their proofs, 
    but does make use of the publicly released blocks.
    """
    public_vals = []
    private_vals = []
    total_hash_cnt = 0
    miners = [ATTACKER,HONEST,HONEST]
    mining = True
    while (mining):
      val= random.getrandbits(self.bits)
      total_hash_cnt+=1

      if val> self.k*self.target:
        continue
      mining_power = [attacker_power,(1-attacker_power)/2,(1-attacker_power)/2]
      miner_idx = np.random.choice(len(mining_power),p=mining_power)

      if miners[miner_idx]=='attacker': #attacking miner
        los = min([x[PROOF] for x in private_vals]+[2**self.bits])
        val_record = [val,miner_idx,los,total_hash_cnt]
        heapq.heappush(private_vals,val_record)
        private_vals=heapq.nsmallest(self.k*8,private_vals)

      else: #miner is not attacking
        los = min([x[PROOF] for x in public_vals]+[2**self.bits])
        val_record = [val,miner_idx,los,total_hash_cnt]
        heapq.heappush(public_vals,val_record)
        heapq.heappush(private_vals,val_record)
        public_vals=heapq.nsmallest(self.k*8,public_vals)
        private_vals=heapq.nsmallest(self.k*8,private_vals)

        public_block= self.check_for_block(public_vals,miner_idx)
        if public_block is not None: #we found a block
          mined_block= self.check_for_block(private_vals,miner_idx)
          if mined_block is None:
            mined_block= self.check_for_block(private_vals,0)
            assert mined_block is not None

          proofs, bonuses = self.calculate_rewards(mined_block,len(miners))
          mining=False
          for idx,strategy in enumerate(miners):
            fd.write("%d, %f,%s, %s, %s, %s, %s\n" % (total_hash_cnt,\
                      mining_power[idx],\
                      "withholding",strategy,
                      proofs[idx],bonuses[idx],self.k))


  def run_withhold_winner_packs(self,attacker_power):
    """ This scenario is a variation of run_withhold() where
    the winning miner packs blocks
    """
    public_vals = []
    private_vals = []
    total_hash_cnt = 0
    miners = [ATTACKER,HONEST,HONEST]
    mining = True
    while (mining):
      val= random.getrandbits(self.bits)
      total_hash_cnt+=1

      if val> self.k*self.target:
        continue
      mining_power = [attacker_power,(1-attacker_power)/2,(1-attacker_power)/2]
      miner_idx = np.random.choice(len(mining_power),p=mining_power)

      if miners[miner_idx]=='attacker': #attacking miner
        los = min([x[PROOF] for x in private_vals]+[2**self.bits])
        val_record = [val,miner_idx,los,total_hash_cnt]
        heapq.heappush(private_vals,val_record)
        private_vals=heapq.nsmallest(self.k*8,private_vals)

      else: #miner is not attacking
        los = min([x[PROOF] for x in public_vals]+[2**self.bits])
        val_record = [val,miner_idx,los,total_hash_cnt]
        heapq.heappush(public_vals,val_record)
        heapq.heappush(private_vals,val_record)
        public_vals=heapq.nsmallest(self.k*8,public_vals)
        private_vals=heapq.nsmallest(self.k*8,private_vals)

        public_block= self.pack_block_miner(public_vals,miner_idx)
        if public_block is not None: #we found a block
          mined_block= self.pack_block_miner(private_vals,miner_idx)
          if mined_block is None:
            assert min(private_vals,key=lambda x:x[PROOF])[MINER]==0
            mined_block = self.pack_block_miner(private_vals,0)
            assert min(mined_block,key=lambda x:x[PROOF])[MINER]==0

          proofs, bonuses = self.calculate_rewards(mined_block,len(miners))
          mining=False
          for idx,strategy in enumerate(miners):
            fd.write("%d, %f,%s, %s, %s, %s, %s\n" % (total_hash_cnt,\
                      mining_power[idx],\
                      "withhding_packed",strategy,
                      proofs[idx],bonuses[idx],self.k))


  def run_withhold_winner_packs_othersFIFO(self,attacker_power):
    """ This scenario is a variation of run_withhold_winner_packs() where
    the winner includes the proofs of other miners in only the order that
    they were received/announced on the network
    """
    public_vals = []
    private_vals = []
    total_hash_cnt = 0
    miners = [ATTACKER,HONEST,HONEST]
    miner_hash_cnt=[0,0,0]
    mining = True
    while (mining):
      val= random.getrandbits(self.bits)
      total_hash_cnt+=1
      assert total_hash_cnt<2**32

      mining_power = [attacker_power,(1-attacker_power)/2,(1-attacker_power)/2]
      miner_idx = np.random.choice(len(mining_power),p=mining_power)
      miner_hash_cnt[miner_idx]+=1

      if val> self.k*self.target:
        continue
      if miners[miner_idx]=='attacker': #attacking miner
        # print('miner found proof',val)
        los = min([x[PROOF] for x in private_vals]+[2**self.bits])
        val_record = [val,miner_idx,los,total_hash_cnt,2**32+total_hash_cnt]
        heapq.heappush(private_vals,val_record)
        private_vals=heapq.nsmallest(self.k*8,private_vals)

      else: #miner is not attacking
        los = min([x[PROOF] for x in public_vals]+[2**self.bits])
        val_record = [val,miner_idx,los,total_hash_cnt,total_hash_cnt]
        heapq.heappush(public_vals,val_record)
        heapq.heappush(private_vals,val_record)
        public_vals=heapq.nsmallest(self.k*8,public_vals)
        private_vals=heapq.nsmallest(self.k*8,private_vals)

        # check if honest has a block
        public_block= self.pack_block_miner_othersFIFO(public_vals,miner_idx)
        if public_block is not None: #honest found a block
          # Assume gamma is 1 and attacker has a chance to release their proofs before
          # this block is put together. attacker's proofs are accepted in fifo order
          mined_block= self.pack_block_miner_othersFIFO(private_vals,miner_idx)

          if mined_block is not None:
            for idx,p in enumerate(public_block):
              assert p[0]==mined_block[idx][0], str(p)+str(mined_block[idx])

          if mined_block is None:
            # this would happen if the original miner now realizes they do not have the 1OS
            # we didn't find a block with private values
            assert min(private_vals,key=lambda x:x[PROOF])[MINER]==0 #must be attacker that has 10s
            # print('\tchecking gamma block for attacker',0)
            mined_block = self.pack_block_miner_othersFIFO(private_vals,0)
            assert mined_block is not None
            assert min(mined_block,key=lambda x:x[PROOF])[MINER]==0

          proofs, bonuses = self.calculate_rewards(mined_block,len(miners))
          mining=False
          author = min(private_vals,key=lambda x:x[PROOF])[MINER] #must be attacker that has 10s

          for idx,strategy in enumerate(miners):
            fd.write("%d, %.2f, %s, %s, %s, %s, %s\n" % (total_hash_cnt, \
                  mining_power[author],\
                  "fifo", strategy,\
                  proofs[0],bonuses[0],self.k))

              
  def pack_block_miner_n(self,n,vals,miner_idx):
    #these are the proofs from this miner
    packer_vals = [x for x in vals if x[MINER]==miner_idx]
    #consider only the k smallest
    packer_vals = heapq.nsmallest(n,packer_vals,key=lambda x:x[PROOF])

    #because we are working on lemma 1, we require n blocks from this FOS miner
    if len(packer_vals)<n:
      return None
    # all acceptable proofs by the other miners
    others_vals = [x for x in vals if x not in packer_vals]
    others_vals =  sorted(others_vals,key=lambda x:x[PROOF])

    block = sorted(packer_vals+others_vals[:self.k-n])
    if len(block)<self.k:
      return None
    if np.mean([x[PROOF] for x in block]) <= self.target:
      return block
    else:
      return None


  def orphan_rate(self,attacker_power,fileid,coin):
    """ 
    Here we calculate the orphan rate of honest miners. We mine a block, and then keep mining 
    until the propagation delay has passed or we have found a second block. 
    """
    if coin=='eth':
      prop_delay = self.theta/15*5 #5  seconds of hash count eth
    elif coin=='bt':
      prop_delay = self.theta/600*10 #10 seconds of hash count for bch
    else:
      exit()
    public_vals = []
    total_hash_cnt = 0
    miners = [HONEST,HONEST]
    mining_power = [attacker_power,(1-attacker_power)]
    winner = None
    # the1os = None

    while (True):
      val= random.getrandbits(self.bits)
      total_hash_cnt+=1

      #don't let this go on forever...
      if winner is not None:
        if (total_hash_cnt-winner_thc ) >= prop_delay:
          return()

      # don't bother if this val isn't going to make it into the block
      if val> self.k*self.target:
        continue
      # determine which miner is author
      miner_idx = np.random.choice(len(mining_power),p=mining_power)
      # create a new proof
      mempool= [x for x in public_vals if (total_hash_cnt-x[TIME]>=prop_delay) or x[MINER]==miner_idx ]
      los = min([x[PROOF] for x in mempool]+[2**self.bits])
      val_record = [val,miner_idx,los,total_hash_cnt]
      heapq.heappush(public_vals,val_record)
      
      mempool= mempool+[val_record]

      public_block= self.check_for_block(mempool,miner_idx)

      if public_block is None: 
        continue
      #we found a block
      proofs, bonuses = self.calculate_rewards(public_block,len(miners))
      if winner is None: #this is the first block
        winner_thc = total_hash_cnt

      for idx,strategy in enumerate(miners):
        #enumerate rewards for all mining pools
        if mining_power[idx]>0:
          fd.write("%d, %f, %d, %d, " % (fileid, attacker_power, self.num_blocks_mined,self.k))
          fd.write("%d, %d, %f,%s, %s, %s\n" % (total_hash_cnt, 
                    total_hash_cnt-winner_thc,mining_power[idx],
                    "first" if winner is None else "second" ,  
                    proofs[idx],bonuses[idx]))
      if winner is not None:
        return()
      else:
        self.num_blocks_mined+=1
        winner = sorted(public_block)[0][MINER]
        if winner==0:
          mining_power = [0,1]
        else:
          mining_power = [1,0]


#############################################



def strategies():
  """ Comparison of strategies, honest to selfish """
  global fd
  fd =open('rewards-%d.csv' % (random.getrandbits(32)),'w')  

  s=sim(k=10)
  for x in range(1000):
    power = random.choice([.1,.2,.3,.4])
    random.choice([\
                    s.run_all_honest(power),\
                    s.run_all_honest_winner_packs(power),\
                    s.run_withhold_winner_packs(power),\
                    s.run_withhold(power),\
                    s.run_withhold_winner_packs_othersFIFO(power)\
                    ])
    fd.flush()
  fd.close()
  
  
def variance():
  """ run a sim of all honest miners just to see the variance of interblock times. """
  global fd
  fd =open('bt-variance-%d.csv' % (random.getrandbits(32)),'w')
  for _ in range(1000):
    #### Comparison of k for HONEST
    k= random.choice([1,5,10,20,30,40])
    s=sim(k=k)
    s.run_all_honest(attacker_power=1)
    fd.flush()


    
def measure_traffic ():
  """ Run a sim to compare trafic """
  global fd, RANDOMNUM
  RANDOMNUM=random.getrandbits(32)
  fd =open('traffic-%d.csv' % (RANDOMNUM),'w')
  for _ in range(1000):
    k=random.randrange(1,80)
    s=sim(k=k)
    s.run_all_honest_count(random.choice([0.999999,0.99999999]))
    fd.flush()
    

def orphan_sim():
  global fd
  # We have to pick a propagation delay and interblock time ahead of time
  # And so we need to select ethereum or bitcoin.
  coin='eth' #or "bt"
  
  RANDOMNUM = random.getrandbits(32) 
  fd= open('%s-orphan-%d.csv' % (coin,RANDOMNUM),'w')

  power=1
  for cnt in range(100):     
    k = random.choice([1,5,10,20,30,40]) 
    s=sim(k)
    s.orphan_rate(power,RANDOMNUM,coin)


# strategies()
# variance()
# orphan_sim()
measure_traffic()