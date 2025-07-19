import numpy as np # type: ignore
import gym # type: ignore
from gym import spaces # type: ignore
import math
import collections
class Scheduler(gym.Env):
    def __init__(self, cqi_embb_user, embb_no=10, urllc_rate=10, ):
        self.TOTAL_EMBB_USERS = embb_no #Total No of EMBB users to schedule
        self.URLLC_RATE = urllc_rate    #URLLC Arrival rate
        self.TOTAL_SLOTS = 100          #No of slots to schedule
        self.TOTAL_MINI_SLOTS = 7       #No of minislots in one slot
        self.TOTAL_PRBS  = 26           #Total available Physical Resource Blocks in one slot
        self.EMBB_ID = np.arange(0,self.TOTAL_EMBB_USERS,dtype=np.int32) #ID of EMBB users
        # self.CQI_TABLE = np.array([2*(120/1024), 
        #                            2*(157/1024), 
        #                            2*(193/1024), 
        #                            2*(251/1024),
        #                            2*(308/1024),
        #                            2*(379/1024),
        #                            2*(449/1024),
        #                            2*(526/1024),
        #                            2*(602/1024),
        #                            2*(679/1024),
        #                            4*(340/1024),
        #                            4*(378/1024),
        #                            4*(434/1024),
        #                            4*(490/1024),
        #                            4*(553/1024),
        #                            4*(616/1024),
        #                            4*(658/1024),
        #                            6*(438/1024),
        #                            6*(466/1024),
        #                            6*(517/1024),
        #                            6*(567/1024), 
        #                            6*(616/1024),
        #                            6*(666/1024),
        #                            6*(719/1024),
        #                            6*(772/1024),
        #                            6*(822/1024),
        #                            6*(873/1024),
        #                            6*(910/1024),
        #                            6*(948/1024)],dtype=np.float32)
        self.CQI_TABLE = np.array([2, 
                                   2, 
                                   2, 
                                   2,
                                   2,
                                   2,
                                   2,
                                   2,
                                   2,
                                   2,
                                   4,
                                   4,
                                   4,
                                   4,
                                   4,
                                   4,
                                   4,
                                   6,
                                   6,
                                   6,
                                   6, 
                                   6,
                                   6,
                                   6,
                                   6,
                                   6,
                                   6,
                                   6,
                                   6],dtype=np.float32)
        self.OH = 0.14                       #Signalling overhead
        self.MIMO_LAYERS = 4                 #No of MIMO Layers
        self.NO_OF_RE_PER_SYM = 12
        self.OFDM_SYM_DURATION = 1/14000     #OFDM Symbol Duration in seconds
        self.OFDM_SYM_DURATION_MS = 1/14     #OFDM Symbol Duration in milliseconds

        # Mutable Variables
        self.final_dr_embb = []
        self.final_dr_urllc = []
        self.cqi_embb_user = cqi_embb_user
        self.Rb_Map = np.zeros((self.TOTAL_SLOTS*self.TOTAL_MINI_SLOTS,self.TOTAL_PRBS,3),
                               dtype=np.int32)
        self.EMBB_buff = np.random.randint(low=70*60*100,high=70*100*100,
                                           size=self.TOTAL_EMBB_USERS) 
        self.Bit_Rate = np.zeros(self.TOTAL_EMBB_USERS,dtype=np.float32)
        self.Total_Data = np.zeros(self.TOTAL_EMBB_USERS,dtype=np.float32)
        self.historical_br = np.ones(self.TOTAL_EMBB_USERS,dtype=np.float32)
        self.inst_data = np.zeros(self.TOTAL_EMBB_USERS,dtype=np.float32)
        self.served_curr_slot = np.zeros(self.TOTAL_EMBB_USERS,dtype=int)
        self.urllc_total_data = 0
        self.urllc_data_rate = 0
        self.urllc_arrival_rate = np.arange(start=5,stop=100,step=5)
        self.punc = 0
        self.total_embb_dr = 0
        self.urllc_ct = 0
        self.embb_prb = 0
        self.extra_urllc = 0
        #RL Variables
        self.action_space = spaces.Discrete(self.TOTAL_PRBS-15)
        self.observation_space = spaces.Box(low=np.array([0,0]),
                                            high=np.array([800,10]))
        self.slots_ep = 0
        self.wr = 0 
        self.total_queue = 0
        self.de = collections.deque([])
        self.de_wait_total = 0
        self.de_wait_avg = 0
        self.urllc_served = 0
        self.de_wait = 0
        self.de_avg_length = 0
        self.normalized_bitrate = 0
        self.normalized_queue_length = 0
        self.queue_length = 0
        self.urllc_served_per_slot = 0

    def queue_slot_arrival(self, arr):
        self.total_queue += arr
        for i in range(len(self.de)):
            self.de[i] += 1
        for i in range(arr):
            self.de.append(0)

    def queue_slot_dep(self, dep):
        for i in range(dep):
            self.de_wait_total += self.de.popleft()
            self.urllc_served += 1
        pass

    def queue_avg_wt(self):
        self.de_wait_avg = self.de_wait_total/self.urllc_served
        return self.de_wait_avg
    
    def get_urllc_latency_stats(self):
        if self.urllc_served > 0:
            avg_latency = self.de_wait_total / self.urllc_served
        else:
            avg_latency = 0
        return avg_latency, self.urllc_served
    
    def queue_avg_length(self, slotid):
        self.de_avg_length = (len(self.de)/(slotid+1))
    
    def queue_wt(self, urllc_ar_index):
        self.de_wait = self.de_avg_length/self.urllc_arrival_rate[urllc_ar_index]
    
    def queue_len(self):
        return len(self.de)

    
    def PFScheduler(self,slot_num, prb):
        ''' 
        ------------------------------------------------------------------------------
        Description:  PF scheduler allocates PRB to different EMBB users based on 
                        their past bitrate information. 
        ------------------------------------------------------------------------------
        Input Arguments:
        RB Map    : Matrix of size=70,50,3
                    RB Map is a numpy array of zeros. The scheduler allocates
                    a PRB to a user, the RB Map is updated with the EMBB User ID.
        Bit Rate  : Array of size=EMBB_USERS
                    This array stores the past bit rate of all the EMBB users.
        Tx Buffer : Array of size=EMBB_USERS
                    This array contains the total data requested by the EMBB users.
        slot_num  : Current time slot of the allocation
        ------------------------------------------------------------------------------
        Output Arguments:
        RB Map    : Updated RB Map after resource block is allocated
        Bit Rate  : ''
        Tx Buffer : ''
        ------------------------------------------------------------------------------
        '''
        embb_user_to_sch = np.argmax(np.divide(self.Bit_Rate,self.historical_br))
        
        self.Rb_Map[slot_num][prb][0] = embb_user_to_sch
        self.Total_Data[embb_user_to_sch] += self.Bit_Rate[embb_user_to_sch]
        self.inst_data[embb_user_to_sch] += self.Bit_Rate[embb_user_to_sch]
        self.historical_br[embb_user_to_sch] = self.Total_Data[embb_user_to_sch]/(slot_num+1)  
        self.EMBB_buff[embb_user_to_sch] -= self.Bit_Rate[embb_user_to_sch]
        self.served_curr_slot[embb_user_to_sch] = 1
        return True

    def update_historical_br(self,slot_num):
        for embb_user in range(0,self.TOTAL_EMBB_USERS):
            if self.served_curr_slot[embb_user] == 0:
                self.historical_br[embb_user] = self.Total_Data[embb_user]/(slot_num+1)
        self.served_curr_slot = np.zeros(self.TOTAL_EMBB_USERS,dtype=int)
        self.urllc_data_rate = self.urllc_total_data/(slot_num+1)
        self.total_embb_dr = sum(self.historical_br)
        #print("EMBB TOTAL DR =",self.total_embb_dr)
        #print("URLLC DATA RATE =",self.urllc_data_rate)
        return True

    def urllc_arrival(self, urllc_ar_index):
        self.urllc_ct = np.random.poisson(lam=self.urllc_arrival_rate[urllc_ar_index])
        print("urllc arrival at n = ", self.urllc_ct)
        self.queue_slot_arrival(self.urllc_ct)
        if self.queue_len() < self.embb_prb:
            self.punc = self.queue_len()
            self.extra_urllc = 0
        else:
            self.extra_urllc = self.queue_len()-self.embb_prb
            self.punc = self.embb_prb
        return True
    
    def puncture_embb(self, slot_no, urllc_alloc_prb):
        '''
        Objective: Puncture the EMBB, i.e. reduce data rate of the users that have 
        been punctured.
        Allocate DR to URLLC DATA RATE VARIABLE
        Recalculate Data rate from respective EMBB users using the RBMap.
        Input: RBMAP, URLLC_DR, EMBB_DR, EMBB_TX_Buff, Puncture_ct
        '''
        #print("\n---------\n Puncture Mode\n----------\n ")
        #print("number of prbs punctured =",self.punc)
        #print('\n')
        urllc_extra_resources = urllc_alloc_prb*7
        self.queue_slot_dep(self.punc)
        print("URLLC DATA             = ", self.urllc_total_data)
        print("urllc extra resources  = ",urllc_extra_resources)
        print("urllc_alloc_prb        = ", urllc_alloc_prb)
        print("queue length           = ",self.queue_len())
        print("extra arrivals         = ",self.extra_urllc)
        print("self.urllc_served      = ",self.urllc_served)
        if urllc_extra_resources > self.queue_len():
            self.queue_slot_dep(self.queue_len())
        else:
            self.queue_slot_dep(urllc_extra_resources)
        
        for prbs in range(0,self.punc):
            self.Total_Data[self.Rb_Map[slot_no][prbs][0]] -= int(self.Bit_Rate[self.Rb_Map[slot_no][prbs][0]]/7)
            self.EMBB_buff[self.Rb_Map[slot_no][prbs][0]] += int(self.Bit_Rate[self.Rb_Map[slot_no][prbs][0]]/7)
        self.urllc_total_data += (1-self.OH)*self.MIMO_LAYERS*self.NO_OF_RE_PER_SYM\
                                 *self.CQI_TABLE[22]*self.punc/self.OFDM_SYM_DURATION_MS/7
        
        
        self.wr = urllc_extra_resources - self.extra_urllc
        self.disc = 0
        if self.wr < 0:
            self.disc = 0-self.wr
            self.wr = 0
            self.urllc_total_data += (1-self.OH)*self.MIMO_LAYERS*self.NO_OF_RE_PER_SYM\
                                 *self.CQI_TABLE[22]*urllc_extra_resources/self.OFDM_SYM_DURATION_MS/7
        else:
            self.urllc_total_data += (1-self.OH)*self.MIMO_LAYERS*self.NO_OF_RE_PER_SYM\
                                 *self.CQI_TABLE[22]*self.extra_urllc/self.OFDM_SYM_DURATION_MS/7
        print("URLLC DATA             = ", self.urllc_total_data)
        print("urllc extra resources  = ",urllc_extra_resources)
        print("urllc_alloc_prb        = ", urllc_alloc_prb)
        print("queue length           = ",self.queue_len())
        print("extra arrivals         = ",self.extra_urllc)
        print("self.urllc_served      = ",self.urllc_served)
        #breakpoint()
        return True

    def update_cqi_br(self):
        self.cqi_embb_user = np.around(np.random.normal(loc=18,scale=5,
                                       size=(self.TOTAL_EMBB_USERS)),decimals=0)
        self.cqi_embb_user = np.clip(self.cqi_embb_user, 0, 28)
        for embbs in range(0,self.TOTAL_EMBB_USERS):
            self.Bit_Rate[embbs] = (1)*self.MIMO_LAYERS*self.NO_OF_RE_PER_SYM\
                                    *self.CQI_TABLE[int(self.cqi_embb_user[embbs])]\
                                    /self.OFDM_SYM_DURATION_MS
            #print("self.CQI = ", self.CQI_TABLE)
            #breakpoint()
        return True
    
    def reset(self):
        self.Rb_Map = np.zeros((self.TOTAL_SLOTS*self.TOTAL_MINI_SLOTS,self.TOTAL_PRBS,3),
                               dtype=np.int32)
        self.EMBB_buff = np.random.randint(low=70*60*100,high=70*100*100,
                                           size=self.TOTAL_EMBB_USERS) 
        #self.Bit_Rate = np.zeros(self.TOTAL_EMBB_USERS,dtype=np.float32)
        self.Total_Data = np.zeros(self.TOTAL_EMBB_USERS,dtype=np.float32)
        self.historical_br = np.ones(self.TOTAL_EMBB_USERS,dtype=np.float32)
        self.served_curr_slot = np.zeros(self.TOTAL_EMBB_USERS,dtype=int)
        self.urllc_total_data = 0
        self.urllc_data_rate = 0
        self.normalized_bitrate = 0
        self.normalized_queue_length = 0
        observation_ = np.array([np.round(self.normalized_bitrate, decimals=2),
                         np.round(self.normalized_queue_length, decimals=2)])
        self.extra_urllc = 0
        self.wr = 0
        self.total_queue = 0
        self.de = collections.deque([])
        self.de_wait_total = 0
        self.de_wait_avg = 0
        self.urllc_served = 0
        self.de_avg_length = 0
        self.de_wait = 0
        self.queue_length = 0

        return observation_

    # def step(self, action):
    #     #print("\nSlot Number",self.slots_ep)
    #     self.inst_data = np.zeros(self.TOTAL_EMBB_USERS,dtype=np.float32)
    #     self.embb_prb = self.TOTAL_PRBS - action
    #     print("action    = ", action)
    #     print("embb prbs = ",self.embb_prb)
    #     for prbs in range(0, self.embb_prb):
    #         self.PFScheduler(slot_num=self.slots_ep,prb=prbs)
    #     self.urllc_arrival(urllc_ar_index=18)
    #     self.puncture_embb(slot_no=self.slots_ep,urllc_alloc_prb=action)
    #     #print("\nEMBB Users served:",self.served_curr_slot)
    #     self.update_historical_br(slot_num=self.slots_ep)
        
    #     #breakpoint()
    #     #print("\nTotal Data: ",self.Total_Data)
    #     #print("\nReq Buffer: ",self.EMBB_buff)
    #     #print("\nHistorical BR: ",self.historical_br)
    #     #print("\nURLLC DR: ",self.urllc_data_rate)

    #     #RL part here
    #     observation_ = np.array([np.round((sum(self.historical_br)/100),decimals=0),
    #                             np.round(self.de_wait_avg, decimals=0)])
    #     print("\nQUEUE WAIT AVG  = ",self.de_wait_avg)
    #     #reward = (1/(1+self.queue_avg_wt()))
    #     #reward = 10+1/(-0.099-10000*np.exp(-10*self.queue_avg_wt()))
        
    #     if sum(self.historical_br)>500 and self.queue_avg_wt()<3:
    #         reward = 20
    #     #elif sum(self.historical_br)<4 and self.queue_avg_wt()>4:
    #     elif sum(self.historical_br)<300 and self.queue_avg_wt()>5:
    #         reward = -90
    #     else:
    #         reward = -1
        
    #     #reward = math.log(sum(self.historical_br)) + 10/(np.exp(0.0001*self.queue_avg_wt()))
    #     self.slots_ep += 1
    #     if self.slots_ep == 40:
    #         self.slots_ep = 0
    #         done = True
    #     else:
    #         done = False
        
    #     return observation_, reward, done
    def step(self, action, ar_index):
        self.inst_data = np.zeros(self.TOTAL_EMBB_USERS, dtype=np.float32)
        self.embb_prb = self.TOTAL_PRBS - action
        for prbs in range(self.embb_prb):
            self.PFScheduler(slot_num=self.slots_ep, prb=prbs)
        self.urllc_arrival(urllc_ar_index=ar_index)
        self.puncture_embb(slot_no=self.slots_ep, urllc_alloc_prb=action)
        self.update_historical_br(slot_num=self.slots_ep)
        self.urllc_served_per_slot = self.urllc_served/(self.slots_ep+1)
        self.queue_length = self.queue_len()
        # Normalize metrics           
        self.normalized_bitrate = sum(self.historical_br) / 80000.0
        self.normalized_queue_length = self.queue_length / 1000.0

        # RL part here
        observation_ = np.array([np.round(self.normalized_bitrate, decimals=2),
                                np.round(self.normalized_queue_length, decimals=2)])

        # Reward Calculation with Dynamic Penalty
        if self.normalized_queue_length > 0.9:
            queue_penalty = 100 * (self.normalized_queue_length - 0.9) ** 2
        else:
            queue_penalty = 0

        reward = self.normalized_bitrate - (0.5 * self.normalized_queue_length) - queue_penalty

        self.slots_ep += 1
        done = self.slots_ep == 40
        if done:
            self.slots_ep = 0

        return observation_, reward, done
