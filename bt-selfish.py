#!/usr/bin/env python3
import random
import numpy as np
import time
import math
import numpy
import heapq

# indices of each val in a proof
PROOF = 0
LOS = 1
DIFF = 500

class mine:
    def __init__(self,k,power):
        self.k=k
        self.power=power
        self.bits=64

        self.vals=[[2**self.bits,2**self.bits]]
        t1=2**self.bits/DIFF
        self.target= float(t1*(self.k+1)/2)
        self.total_bonuses=0

    def should_mine(self):
        return random.random() < self.power

    def val_mean(self):
        return (sum(self.vals)/len(self.vals))

    def check_success(self):
        #mine a new proof
        val = random.getrandbits(self.bits)
        los = min([x[PROOF] for x in self.vals]+[2**self.bits])        
        val_record=[val,los]
        self.vals= heapq.nsmallest(self.k,self.vals+[val_record],key=lambda x: x[PROOF])
        if np.mean([float(x[PROOF]) for x in self.vals])<self.target:
            # print()
            return(True)
        else:
            return(False)


def selfish_trial(gamma,q,k):
    acnt=0
    hcnt=0
    honest_length = 0
    attacker_length = 0
    honest = mine(k,max(0,1-q))
    attacker= mine(k,q)    
    hon_first = mine(k,max(0,1-q))
    att_first= mine(k,q)
    
    while (True):
        att_won =False
        hon_won =False
        att_reward = 0
        att_bonus = 0
        
        dice = np.random.uniform(0,1)
        if (dice<q):
          if(attacker_length==0):
              val = random.getrandbits(att_first.bits)

              #do the best we can with the LOS
              #find the min honest proof of the ones that are larger than the current attacker's min
              att_min_val = min([x[PROOF] for x in att_first.vals])
              hon_los = min([x[PROOF] for x in hon_first.vals if x[PROOF]>att_min_val]+[2**att_first.bits])
                
              #now add to the attacker's list of proof  
              val_record=[val,hon_los]
              att_first.vals=heapq.nsmallest(k,att_first.vals+[val_record],key=lambda x: x[PROOF])

              #to test for a block, add in all honest that have an LOS that is larger than attacker min, 
              #and where the proof is larger than the attacker's min
              att_min_val = min([x[PROOF] for x in att_first.vals])
              hon_vals = [x for x in hon_first.vals if x[PROOF]>att_min_val and x[LOS]>att_min_val]
              block= heapq.nsmallest(k,att_first.vals+hon_vals,key=lambda x: x[PROOF])
              if np.mean([float(x[PROOF]) for x in block])<att_first.target:
                att_won=True
                # calculate rewards from proofs; honest is 1 minus this value
                att_fork_att_reward = sum([x in att_first.vals for x in block])/float(k)
                # calculate bonuses
                block_1os = min([x[PROOF] for x in block])
                att_bonus = sum([((x in att_first.vals) and (x[LOS]==block_1os))  for x in block])/float(k)
                hon_bonus = sum([((x in hon_first.vals) and (x[LOS]==block))  for x in block])/float(k)

                
          else:
            if attacker.check_success():
              att_won=True
              #calculate bonuses
              FOS = min([x[PROOF] for x in attacker.vals])
              attacker.total_bonuses  += sum([x[LOS]==FOS for x in attacker.vals])/float(k)
              #reset 
              attacker.vals=[[2**attacker.bits,2**attacker.bits]]

        else:
          if(honest_length==0):
            #add a proof to honest vals
            val = random.getrandbits(hon_first.bits)
            los = min([x[PROOF] for x in hon_first.vals]+[2**hon_first.bits])
            
            val_record=[val,los]
            # We are assuming that the attacker cant release withheld proofs until they know the 
            # new lowest honest OS. So when creating the block, the honest include only
            # those attacker proofs that were previously relased.
            # So first we figure out what was released
            # then we update the list of k-lowest honest att_proofs
            # then we try to create a block 
            
            # add in the attacker proofs, if they are larger than the honest 1os
            hon_min_val = min([x[PROOF] for x in hon_first.vals])
            att_vals = [x for x in att_first.vals if x[PROOF]>hon_min_val]

            # keep only the k smallest, now including the new val_record
            hon_first.vals= heapq.nsmallest(k,hon_first.vals+[val_record],key=lambda x: x[PROOF])
            block= heapq.nsmallest(k,hon_first.vals+att_vals,key=lambda x: x[PROOF])
            
            if np.mean([float(x[PROOF]) for x in block])<hon_first.target:
                hon_won=True
                #compute rewards based on block's 1os
                block_1os = min([x[PROOF] for x in block])
                #attacker's reward; honest is 1 minus this value
                hon_fork_att_reward = sum([x in att_first.vals for x in block])/float(k)
                #bonuses, if any
                att_bonus = sum([((x in att_first.vals) and (x[LOS]==block_1os))  for x in block])/float(k)
                hon_bonus = sum([((x in hon_first.vals) and (x[LOS]==block_1os))  for x in block])/float(k)

          else:
            if honest.check_success():
              hon_won=True          
              #calculate bonuses    
              FOS = min([x[PROOF] for x in honest.vals])
              honest.total_bonuses  += sum([x[LOS]==FOS for x in honest.vals])/float(k)
              #reset
              honest.vals=[[2**honest.bits,2**honest.bits]]
              
        #Original SM algorithm
        if att_won:
            #attacker node wins block
            delta_prev = attacker_length - honest_length
            attacker_length+=1
            if (delta_prev == 0 and attacker_length>=2):
                #attack breaks tie in her favor
                return(attacker_length,0,
                (attacker_length-1)+att_fork_att_reward,
                1-att_fork_att_reward,
                att_bonus,
                attacker.total_bonuses,
                hon_bonus,
                0)
                
        if hon_won:
            #honest node wins block
            delta_prev = attacker_length - honest_length
            honest_length += 1
            if (delta_prev == 0):
                if (attacker_length==0):
                    #honest won right out of the gate
                    return(0,honest_length,
                    hon_fork_att_reward,
                    (honest_length-1)+(1-hon_fork_att_reward),
                    att_bonus,
                    0,
                    hon_bonus,
                    honest.total_bonuses
                    )
                else:
                    # we were previously in a tie, but honest won
                    # check *which fraction* of the honest nodes won
                    dice2 = np.random.uniform(0,1)

                    if (dice2 < (gamma)):
                        #honest added to attacker fork; game over
                        #special case. figure out bonsues of last block only
                        FOS = min([x[PROOF] for x in honest.vals])
                        last_bonus  = sum([x[LOS]==FOS for x in honest.vals])/float(k)
                        
                        return(attacker_length,1,
                        (attacker_length-1)+att_fork_att_reward,
                        1+1-att_fork_att_reward,
                        att_bonus,
                        attacker.total_bonuses,
                        hon_bonus,
                        last_bonus
                        )
                    else:
                        # honest added to its own fork; game over
                        return(0, honest_length,
                        hon_fork_att_reward, 
                        (honest_length-1)+(1-hon_fork_att_reward),
                        att_bonus,
                        0,
                        hon_bonus,
                        honest.total_bonuses
                        )
            elif (delta_prev==1):
                # now we are tied.
                # keep mining to see who wins; we check gamma later
                pass
            elif (delta_prev==2):
                # attacker bails because, after this latest win, it's only 1 ahead
                return(attacker_length,0,
                (attacker_length-1)+att_fork_att_reward,
                1-att_fork_att_reward,
                att_bonus,
                attacker.total_bonuses,
                hon_bonus,
                0
                )



def selfish_sim(gamma,q,k):
  props=[]

  hon_wins_blks=0
  att_wins_blks=0  
  hon_wins=0
  att_wins=0
  att_bonus_f=0
  hon_bonus_f=0
  att_bonus_c=0
  hon_bonus_c=0
  
  while (att_wins_blks+hon_wins_blks<(1000)):
    attacker_win_blks,honest_wins_blks,attacker_win,honest_wins,attacker_bonus_first,a_b_cdr,honest_bonus_first,h_b_cdr=selfish_trial(gamma,q,k)
    att_wins+=attacker_win
    hon_wins+=honest_wins
    att_wins_blks += attacker_win_blks
    hon_wins_blks += honest_wins_blks
    att_bonus_f+=attacker_bonus_first
    hon_bonus_f+=honest_bonus_first    
    att_bonus_c+=a_b_cdr
    hon_bonus_c+=h_b_cdr
    

  return([att_wins_blks,hon_wins_blks,att_wins,hon_wins,att_bonus_f,hon_bonus_f,att_bonus_c,hon_bonus_c])


t=int(time.time())
r=np.random.randint(0,100000) 
with open('revSM.rand=%d%d.csv' % (t,r), 'w') as fd:

  for i in range(0,20):
    k=random.choice([1,5,20,100])
    gamma= random.choice([0,1])    
    q=random.choice([.1,.2,.3,.4,.45])
    fd.write("%s,%s,%s,%s," %(DIFF,q,k,gamma))
    fd.flush()
    a_blks,h_blks,att_proofs,hon_proofs,a_bonus_f,h_bonus_f,a_b_c,h_b_c =selfish_sim(gamma,q,k)
    fd.write("%s,%s,%s,%s,%s,%s,%s,%s\n" %(att_proofs,hon_proofs,a_bonus_f,h_bonus_f,a_b_c,h_b_c,a_blks,h_blks))
    fd.flush()

