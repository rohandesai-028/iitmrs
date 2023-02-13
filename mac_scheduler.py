import numpy as np
import gym
from gym import spaces
class Scheduler():
    def __init__(self, cqi_embb_user, embb_no=10, urllc_rate=10, ):
        self.TOTAL_EMBB_USERS = embb_no #Total No of EMBB users to schedule
        self.URLLC_RATE = urllc_rate    #URLLC Arrival rate
        self.TOTAL_SLOTS = 100          #No of slots to schedule
        self.TOTAL_MINI_SLOTS = 7       #No of minislots in one slot
        self.TOTAL_PRBS  = 26           #Total available Physical Resource Blocks in one slot
        self.EMBB_ID = np.arange(0,self.TOTAL_EMBB_USERS,dtype=np.int32) #ID of EMBB users
        self.CQI_TABLE = np.array([2*(120/1024), 
                                2*(157/1024), 
                                2*(193/1024), 
                                2*(251/1024),
                                2*(308/1024),
                                2*(379/1024),
                                2*(449/1024),
                                2*(526/1024),
                                2*(602/1024),
                                2*(679/1024),
                                4*(340/1024),
                                4*(378/1024),
                                4*(434/1024),
                                4*(490/1024),
                                4*(553/1024),
                                4*(616/1024),
                                4*(658/1024),
                                6*(438/1024),
                                6*(466/1024),
                                6*(517/1024),
                                6*(567/1024), 
                                6*(616/1024),
                                6*(666/1024),
                                6*(719/1024),
                                6*(772/1024),
                                6*(822/1024),
                                6*(873/1024),
                                6*(910/1024),
                                6*(948/1024)],dtype=np.float32)
        self.OH = 0.14                       #Signalling overhead
        self.MIMO_LAYERS = 4                 #No of MIMO Layers
        self.NO_OF_SYM_PER_RB = 12
        self.OFDM_SYM_DURATION = 1/14000     #OFDM Symbol Duration in seconds
        self.OFDM_SYM_DURATION_MS = 1/14     #OFDM Symbol Duration in milliseconds

        # Mutable Variables
        self.final_dr_embb = []
        self.final_dr_urllc = []
        self.cqi_embb_user = cqi_embb_user
        self.Rb_Map = np.zeros((self.TOTAL_SLOTS*self.TOTAL_MINI_SLOTS,self.TOTAL_PRBS,3),dtype=np.int32)
        self.EMBB_buff = np.random.randint(low=70*60*100,high=70*100*100,size=self.TOTAL_EMBB_USERS) 
        self.Bit_Rate = np.zeros(self.TOTAL_EMBB_USERS,dtype=np.float32)
        self.Total_Data = np.zeros(self.TOTAL_EMBB_USERS,dtype=np.float32)
        self.historical_br = np.ones(self.TOTAL_EMBB_USERS,dtype=np.float32)
        self.served_curr_slot = np.zeros(self.TOTAL_EMBB_USERS,dtype=int)
        self.urllc_total_data = 0
        self.urllc_data_rate = 0
        self.urllc_arrival_rate = np.arange(start=5,stop=60,step=5)
        self.punc = 0
        self.total_embb_dr = 0
        self.urllc_ct = 0
        #RL Variables
        self.action_space = spaces.Discrete(self.TOTAL_PRBS)
        self.observation_space = spaces.Box(low=np.array(0,0,0),high=np.array(100000,200,100000))

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
        print("EMBB TOTAL DR =",self.total_embb_dr)
        print("URLLC DATA RATE =",self.urllc_data_rate)
        return True

    def urllc_arrival(self, urllc_ar_index):
        self.urllc_ct = np.random.poisson(lam=self.urllc_arrival_rate[urllc_ar_index])
        if self.urllc_ct < self.TOTAL_PRBS:
            self.punc = self.urllc_ct
        else:
            self.punc = self.TOTAL_PRBS
        return True
    
    def puncture_embb(self, slot_no):
        '''
        Objective: Puncture the EMBB, i.e. reduce data rate of the users that have 
        been punctured.
        Allocate DR to URLLC DATA RATE VARIABLE
        Recalculate Data rate from respective EMBB users using the RBMap.
        Input: RBMAP, URLLC_DR, EMBB_DR, EMBB_TX_Buff, Puncture_ct
        '''
        print("\n---------\n Puncture Mode\n----------\n ")
        print("number of prbs punctured =",self.punc)
        print('\n')
        for prbs in range(0,self.punc):
            self.Total_Data[self.Rb_Map[slot_no][prbs][0]] -= int(self.Bit_Rate[self.Rb_Map[slot_no][prbs][0]]/7)
            self.EMBB_buff[self.Rb_Map[slot_no][prbs][0]] += int(self.Bit_Rate[self.Rb_Map[slot_no][prbs][0]]/7)
        self.urllc_total_data += (1-self.OH)*self.MIMO_LAYERS*self.NO_OF_SYM_PER_RB\
                                 *self.CQI_TABLE[22]*self.punc/self.OFDM_SYM_DURATION_MS/7
        print("URLLC DATA =", self.urllc_total_data)
        return True

    def update_cqi_br(self):
        #self.cqi_embb_user = np.around(np.random.normal(loc=18,scale=5,
        #                               size=(self.TOTAL_EMBB_USERS)),decimals=0)
        #self.cqi_embb_user = np.clip(self.cqi_embb_user, 0, 28)
        for embbs in range(0,self.TOTAL_EMBB_USERS):
            self.Bit_Rate[embbs] = (1-self.OH)*self.MIMO_LAYERS*self.NO_OF_SYM_PER_RB\
                                    *self.CQI_TABLE[int(self.cqi_embb_user[embbs])]\
                                    /self.OFDM_SYM_DURATION_MS
        return True
    
    def scheduler_reset(self):
        self.Rb_Map = np.zeros((self.TOTAL_SLOTS*self.TOTAL_MINI_SLOTS,self.TOTAL_PRBS,3),dtype=np.int32)
        self.EMBB_buff = np.random.randint(low=70*60*100,high=70*100*100,size=self.TOTAL_EMBB_USERS) 
        #self.Bit_Rate = np.zeros(self.TOTAL_EMBB_USERS,dtype=np.float32)
        self.Total_Data = np.zeros(self.TOTAL_EMBB_USERS,dtype=np.float32)
        self.historical_br = np.ones(self.TOTAL_EMBB_USERS,dtype=np.float32)
        self.served_curr_slot = np.zeros(self.TOTAL_EMBB_USERS,dtype=int)
        self.urllc_total_data = 0
        self.urllc_data_rate = 0

    def scheduler_step(self, slot_no, rate, action):
        print("\nSlot Number",slot_no)
        for prbs in range(0, self.TOTAL_PRBS):
            self.PFScheduler(slot_num=slot_no,prb=prbs)
        self.urllc_arrival(urllc_ar_index=rate)
        self.puncture_embb(slot_no=slot_no)
        print("\nEMBB Users served:",self.served_curr_slot)
        self.update_historical_br(slot_num=slot_no)
        print("\nTotal Data: ",self.Total_Data)
        print("\nReq Buffer: ",self.EMBB_buff)
        print("\nHistorical BR: ",self.historical_br)
        print("\nURLLC DR: ",self.urllc_data_rate)
