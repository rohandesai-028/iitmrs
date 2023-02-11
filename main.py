import numpy as np
import matplotlib.pyplot as plt
from gym import Env
from gym.spaces import Discrete
from gym.spaces import Box
print("\n\nHello, I'm a scheduler!!! :)\n\n")
# -----------------------------------------------------------------------
# Variable Declarations
# -----------------------------------------------------------------------
TOTAL_SLOTS = 100         #No of slots to schedule
TOTAL_MINI_SLOTS = 7     #No of minislots in one slot
TOTAL_PRBS  = 26        #Total available Physical Resource Blocks in one slot
TOTAL_EMBB_USERS = 10   #Total No of EMBB users to schedule
URLLC_RATE = 5
EMBB_ID = np.arange(0,TOTAL_EMBB_USERS,dtype=np.int32) #ID of EMBB users
CQI_TABLE = np.array([2*(120/1024), 
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
OH = 0.14                       #Signalling overhead
MIMO_LAYERS = 4                 #No of MIMO Layers
NO_OF_SYM_PER_RB = 12
OFDM_SYM_DURATION = 1/14000     #OFDM Symbol Duration in seconds
OFDM_SYM_DURATION_MS = 1/14     #OFDM Symbol Duration in milliseconds
EPISODE_DURATION = 10           #number of actions in one episode(10 slots)




final_dr_embb = []
final_dr_urllc = []
cqi_embb_user = np.around(np.random.normal(loc=18,scale=5,size=(TOTAL_EMBB_USERS)),decimals=0)
cqi_embb_user = np.clip(cqi_embb_user, 0, 28)
Rb_Map = np.zeros((TOTAL_SLOTS*TOTAL_MINI_SLOTS,TOTAL_PRBS,3),dtype=np.int32)
EMBB_buff = np.random.randint(low=70*60*100,high=70*100*100,size=TOTAL_EMBB_USERS) 
Bit_Rate = np.zeros(TOTAL_EMBB_USERS,dtype=np.float32)
Total_Data = np.zeros(TOTAL_EMBB_USERS,dtype=np.float32)
historical_br = np.ones(TOTAL_EMBB_USERS,dtype=np.float32)
served_curr_slot = np.zeros(TOTAL_EMBB_USERS,dtype=int)
urllc_total_data = 0
urllc_data_rate = 0
urllc_arrival_rate = np.arange(start=5,stop=80,step=3)
# -----------------------------------------------------------------------
# Function Definitions
# -----------------------------------------------------------------------
def PFScheduler(slot_num, prb, Rb_Map, EMBB_buff, Bit_Rate, Total_Data, historical_br):
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
    embb_user_to_sch = np.argmax(np.divide(Bit_Rate,historical_br))
    
    Rb_Map[slot_num][prb][0] = embb_user_to_sch
    Total_Data[embb_user_to_sch] += Bit_Rate[embb_user_to_sch]
    historical_br[embb_user_to_sch] = Total_Data[embb_user_to_sch]/(slot_num+1)  
    EMBB_buff[embb_user_to_sch] -= Bit_Rate[embb_user_to_sch]
    served_curr_slot[embb_user_to_sch] = 1
#--------------------------------------------------------------------------------
def update_historical_br(served, br, TOTAL_EMBB_USERS, Total, slot_num,urllc_total):
    for embb_user in range(0,TOTAL_EMBB_USERS):
        if served[embb_user] == 0:
            br[embb_user] = Total[embb_user]/(slot_num+1)
    served = np.zeros(TOTAL_EMBB_USERS,dtype=int)
    urllc_dr = urllc_total/(slot_num+1)
    total_embb_dr = sum(br)
    print("EMBB TOTAL DR =",total_embb_dr)
    print("URLLC DATA RATE =",urllc_data_rate)
    return served, br, urllc_dr, total_embb_dr
#--------------------------------------------------------------------------------
def puncture_embb(no_of_punc, Rb_Map, slot_no, embb_total_data, embb_br, embb_tx, urllc_total):
    '''
    Objective: Puncture the EMBB, i.e. reduce data rate of the users that have 
    been punctured.
    Allocate DR to URLLC DATA RATE VARIABLE
    Recalculate Data rate from respective EMBB users using the RBMap.

    Input: RBMAP, URLLC_DR, EMBB_DR, EMBB_TX_Buff, Puncture_ct

    '''
    print("\n---------\n Puncture Mode\n----------\n ")
    print("number of prbs punctured =",no_of_punc)
    print('\n')
    global OH, MIMO_LAYERS, NO_OF_SYM_PER_RB, CQI_TABLE, OFDM_SYM_DURATION_MS
    for prbs in range(0,no_of_punc):
        embb_total_data[Rb_Map[slot_no][prbs][0]] -= int(embb_br[Rb_Map[slot_no][prbs][0]]/7)
        embb_tx[Rb_Map[slot_no][prbs][0]] += int(embb_br[Rb_Map[slot_no][prbs][0]]/7)
    urllc_total += (1-OH)*MIMO_LAYERS*NO_OF_SYM_PER_RB*CQI_TABLE[22]*no_of_punc/OFDM_SYM_DURATION_MS/7
    print("URLLC DATA =", urllc_total)
    return urllc_total
    pass
#--------------------------------------------------------------------------------
def update_br():
    global Bit_Rate, TOTAL_EMBB_USERS, MIMO_LAYERS, NO_OF_SYM_PER_RB, CQI_TABLE, OFDM_SYM_DURATION_MS
    cqi_embb_user = np.around(np.random.normal(loc=18,scale=5,size=(TOTAL_EMBB_USERS)),decimals=0)
    cqi_embb_user = np.clip(cqi_embb_user, 0, 28)
    for embbs in range(0,TOTAL_EMBB_USERS):
        Bit_Rate[embbs] = (1-OH)*MIMO_LAYERS*NO_OF_SYM_PER_RB*CQI_TABLE[int(cqi_embb_user[embbs])]/OFDM_SYM_DURATION_MS

    pass
#--------------------------------------------------------------------------------
def urllc_arrival(urllc_arrival_rate):
    embb_punct_ct = np.random.poisson(lam=urllc_arrival_rate)
    if embb_punct_ct < TOTAL_PRBS:
        punc = embb_punct_ct
    else:
        punc = TOTAL_PRBS
    return punc
    pass
#--------------------------------------------------------------------------------
print("\nScheduler Start")
print("Number of EMBB Users", TOTAL_EMBB_USERS)
print("CQI of EMBB Users", cqi_embb_user)
print("EMBB Buffer data", EMBB_buff)
#print("RB map", Rb_Map)
cqi_embb_user = np.around(np.random.normal(loc=18,scale=5,size=(TOTAL_EMBB_USERS)),decimals=0)
cqi_embb_user = np.clip(cqi_embb_user, 0, 28)
for i in range(0,len(urllc_arrival_rate)):
    
    Rb_Map = np.zeros((TOTAL_SLOTS*TOTAL_MINI_SLOTS,TOTAL_PRBS,3),dtype=np.int32)
    EMBB_buff = np.random.randint(low=70*60*100,high=70*100*100,size=TOTAL_EMBB_USERS) 
    Bit_Rate = np.zeros(TOTAL_EMBB_USERS,dtype=np.float32)
    Total_Data = np.zeros(TOTAL_EMBB_USERS,dtype=np.float32)
    historical_br = np.ones(TOTAL_EMBB_USERS,dtype=np.float32)
    served_curr_slot = np.zeros(TOTAL_EMBB_USERS,dtype=int)
    urllc_total_data = 0
    urllc_data_rate = 0
    for embbs in range(0,TOTAL_EMBB_USERS):
        Bit_Rate[embbs] = (1-OH)*MIMO_LAYERS*NO_OF_SYM_PER_RB*CQI_TABLE[int(cqi_embb_user[embbs])]/OFDM_SYM_DURATION_MS

    print("Bit Rate of the EMBB users",Bit_Rate)
    print("Historical Bit Rate",historical_br)
    for slots in range(0,TOTAL_SLOTS):
        print("\n\nSlot Number",slots)
        for prb in range(0,TOTAL_PRBS):
            PFScheduler(slots, prb, Rb_Map, EMBB_buff, Bit_Rate, Total_Data, 
                        historical_br)
        punc_ct = urllc_arrival(urllc_arrival_rate=urllc_arrival_rate[i])
        urllc_total_data = puncture_embb(punc_ct, Rb_Map, slots,Total_Data,
                    Bit_Rate, EMBB_buff,urllc_total_data)
        print(Rb_Map[slots])
        print("EMBB Users served:",served_curr_slot)
        served_curr_slot, historical_br, urllc_data_rate, total_br = update_historical_br(served_curr_slot, 
                                                                historical_br, 
                                                                TOTAL_EMBB_USERS,
                                                                Total_Data, 
                                                                slots, 
                                                                urllc_total_data)
        print("Total Data:",Total_Data)
        print("Req Buffer:",EMBB_buff)
        print("Historical BR:",historical_br)
        print("URLLC Data Rate: ",urllc_data_rate)

    final_dr_urllc.append(urllc_data_rate)
    final_dr_embb.append(total_br)

print("EMBB DR for different arrival rates",final_dr_embb)
print("URLLC DR for different arrival rates",final_dr_urllc)
print(urllc_arrival_rate)
plt.plot(urllc_arrival_rate,final_dr_embb)
plt.show()
#--------------------------------------------------------------------------------
# Template environment setup
#--------------------------------------------------------------------------------
NUMBER_OF_ACTIONS = TOTAL_PRBS
class RB_ENV(Env):
    def __init__(self):
        self.action_space = Discrete(NUMBER_OF_ACTIONS)
        #No of RBs to be handed over to urllc in one slot
        #List of all the parameters in observation space 
        #with their lows and highs
        #URLLC Arrival Rate: [0,10000]
        #EMBB USER Buffer data Total: [0, 1000000]
        #EMBB Total Data Rate: [0,500000]
        #URLLC Total Data Rate:[0,500000]
        #EMBB CQI:[0,28]
        
        self.observation_space = Box(low=np.array(0,0,0,0), high=np.array(100000,10000000,5000000,5000000), dtype=np.float32)

    
        pass
    def step(self):
        #defines what we do at every state
        pass
    def render(self):
        #dont do anything
        pass
    def reset(self):
        #reset environment after episode
        pass

